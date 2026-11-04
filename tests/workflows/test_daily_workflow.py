import logging

import numpy as np
import pandas as pd
import pytest

from optpricing.calibration import CalibrationSettings, LiquidityFilter
from optpricing.models import BSMModel, HestonModel, KouModel, MertonJumpModel
from optpricing.workflows import DailyWorkflow
from optpricing.workflows.configs import ALL_MODEL_CONFIGS

FAST = CalibrationSettings(n_starts=1)


def test_success_path_recovers_bsm_sigma_offline(bsm_snapshot, no_network):
    """Market parameters come from the snapshot, no yfinance call is made."""
    cfg = {**ALL_MODEL_CONFIGS["BSM"], "ticker": "TEST"}
    wf = DailyWorkflow(bsm_snapshot, cfg, settings=FAST)
    wf.run()
    r = wf.results
    assert r["Status"] == "Success", r.get("Error")
    assert r["Ticker"] == "TEST" and r["Model"] == "BSM"
    assert r["Calibrated Params"]["sigma"] == pytest.approx(0.23, abs=1e-4)
    assert r["Implied Rate"] == pytest.approx(0.04)
    assert r["Known Dividend"] == pytest.approx(0.01)
    assert r["RMSE"] < 1e-6 and r["IV RMSE"] < 1e-5
    assert (
        r["N Calibration Options"] > 0 and r["N Options"] >= r["N Calibration Options"]
    )
    assert isinstance(r["Buckets"], pd.DataFrame)
    assert wf.stock.dividend == 0.01 and wf.rate.get_rate() == 0.04
    assert isinstance(wf.model, BSMModel)


def test_ticker_taken_from_snapshot_when_missing_in_recipe(bsm_snapshot):
    wf = DailyWorkflow(bsm_snapshot, dict(ALL_MODEL_CONFIGS["BSM"]), settings=FAST)
    assert wf.results["Ticker"] == "TEST"


def test_v1_snapshot_without_market_columns_still_runs(bsm_snapshot, no_network):
    """Legacy frames (no r/q columns): rate defaults, q implied from parity."""
    legacy = bsm_snapshot.drop(
        columns=["ticker", "snapshot_date", "as_of", "risk_free_rate", "dividend_yield"]
    )
    wf = DailyWorkflow(
        legacy, {**ALL_MODEL_CONFIGS["BSM"], "ticker": "TEST"}, settings=FAST
    )
    wf.run()
    assert wf.results["Status"] == "Success", wf.results.get("Error")
    assert wf.results["Implied Rate"] == pytest.approx(0.04)
    assert wf.results["Known Dividend"] == pytest.approx(0.01, abs=2e-3)


def test_recipe_bounds_take_precedence_over_param_defs(bsm_snapshot):
    """Regression: `param_defs` used to override the recipe (Bates even fitted a
    non-existent `sigma`). A recipe bound that excludes the true vol must bind."""
    cfg = {
        **ALL_MODEL_CONFIGS["BSM"],
        "bounds": {"sigma": (0.3, 0.5)},
        "initial_guess": {"sigma": 0.4},
    }
    wf = DailyWorkflow(bsm_snapshot, cfg, settings=FAST)
    wf.run()
    assert wf.results["Status"] == "Success"
    assert wf.results["Calibrated Params"]["sigma"] == pytest.approx(0.3, abs=1e-6)


def test_kou_recipe_is_calibratable(synthetic_snapshot, no_network):
    """Regression: the Kou recipe froze a *list* of names and crashed."""
    truth = KouModel(
        params={"sigma": 0.15, "lambda": 1.0, "p_up": 0.3, "eta1": 12.0, "eta2": 6.0}
    )
    chain = synthetic_snapshot(truth)
    wf = DailyWorkflow(
        chain, {**ALL_MODEL_CONFIGS["Kou"], "ticker": "TEST"}, settings=FAST
    )
    wf.run()
    assert wf.results["Status"] == "Success", wf.results.get("Error")
    assert wf.results["Calibration IV RMSE"] < 0.01
    assert set(wf.results["Calibrated Params"]) == {
        "sigma",
        "lambda",
        "p_up",
        "eta1",
        "eta2",
    }


def test_merton_fits_all_parameters_with_historical_prior(
    synthetic_snapshot, monkeypatch, no_network
):
    """Regression: the historical strategy froze every Merton parameter."""
    truth = MertonJumpModel(
        params={
            "sigma": 0.15,
            "lambda": 0.6,
            "mu_j": -0.15,
            "sigma_j": 0.2,
            "max_sum_terms": 100,
        }
    )
    chain = synthetic_snapshot(truth)
    returns = pd.Series(np.random.default_rng(0).standard_t(3, 2000) * 0.01)
    monkeypatch.setattr(
        "optpricing.workflows.daily_workflow.historical_returns_if_available",
        lambda ticker, period="10y": returns,
    )
    wf = DailyWorkflow(
        chain, {**ALL_MODEL_CONFIGS["Merton"], "ticker": "TEST"}, settings=FAST
    )
    wf.run()
    assert wf.results["Status"] == "Success", wf.results.get("Error")
    params = wf.results["Calibrated Params"]
    assert params["sigma"] == pytest.approx(0.15, abs=0.02)
    assert wf.results["Calibration IV RMSE"] < 5e-3


def test_legacy_list_frozen_is_ignored(bsm_snapshot):
    cfg = {**ALL_MODEL_CONFIGS["BSM"], "frozen": ["sigma"]}
    wf = DailyWorkflow(bsm_snapshot, cfg, settings=FAST)
    wf.run()
    assert wf.results["Status"] == "Success"
    assert wf.results["Calibrated Params"]["sigma"] == pytest.approx(0.23, abs=1e-4)


def test_settings_control_the_calibration_set(bsm_snapshot):
    tight = CalibrationSettings(
        liquidity=LiquidityFilter(moneyness=(0.95, 1.05), min_maturity_days=60),
        n_starts=1,
    )
    wf = DailyWorkflow(bsm_snapshot, dict(ALL_MODEL_CONFIGS["BSM"]), settings=tight)
    wf.run()
    assert wf.results["Status"] == "Success"
    assert wf.calibration_data["moneyness"].between(0.95, 1.05).all()
    assert (wf.calibration_data["maturity"] >= 60 / 365.25).all()
    assert len(wf.evaluation_data) > len(wf.calibration_data)


def test_mc_only_model_fails_cleanly_without_allow_mc(bsm_snapshot, caplog):
    cfg = {**ALL_MODEL_CONFIGS["SABRJump"], "ticker": "TEST"}
    with caplog.at_level(logging.ERROR):
        wf = DailyWorkflow(bsm_snapshot, cfg, settings=FAST)
        wf.run()
    assert wf.results["Status"] == "Failed"
    assert "Monte Carlo" in wf.results["Error"]
    assert np.isnan(wf.results["RMSE"])
    assert "Workflow failed" in caplog.text


def test_failure_path_records_error(bsm_snapshot):
    too_few = bsm_snapshot.head(4)
    wf = DailyWorkflow(too_few, dict(ALL_MODEL_CONFIGS["BSM"]), settings=FAST)
    wf.run()
    assert wf.results["Status"] == "Failed"
    assert "cannot calibrate" in wf.results["Error"]
    assert "Runtime (s)" in wf.results


def test_warm_start_is_used_as_a_start_point(synthetic_snapshot):
    truth = HestonModel(
        params={"v0": 0.05, "kappa": 1.5, "theta": 0.06, "rho": -0.6, "vol_of_vol": 0.5}
    )
    chain = synthetic_snapshot(truth)
    cfg = {**ALL_MODEL_CONFIGS["Heston"], "ticker": "TEST"}
    wf = DailyWorkflow(chain, cfg, settings=FAST, warm_start=dict(truth.params))
    wf.run()
    assert wf.results["Status"] == "Success"
    assert any(s["kind"] == "warm" for s in wf.calibrator.result.starts)
    assert wf.results["Calibration IV RMSE"] < 1e-3


def test_evaluate_on_other_data(bsm_snapshot, synthetic_snapshot):
    wf = DailyWorkflow(bsm_snapshot, dict(ALL_MODEL_CONFIGS["BSM"]), settings=FAST)
    wf.run()
    other = synthetic_snapshot(BSMModel({"sigma": 0.3}))
    summary, buckets, residuals = wf.evaluate(wf.model, data=other)
    assert summary["rmse_iv"] == pytest.approx(0.07, abs=2e-3)
    assert len(residuals) == len(other)
    assert wf._evaluate_rmse(wf.model, wf.stock, wf.rate) < 1e-6
