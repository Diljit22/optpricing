import json

import pandas as pd
import pytest

from optpricing.calibration import CalibrationSettings
from optpricing.data import market_data_manager
from optpricing.models import BSMModel
from optpricing.workflows import BacktestWorkflow, backtest_workflow
from optpricing.workflows.configs import ALL_MODEL_CONFIGS

FAST = CalibrationSettings(n_starts=1)


@pytest.fixture
def corpus(tmp_path, monkeypatch, synthetic_snapshot):
    """Three snapshot days with different generating vols, stored newest-first
    on disk order to make sure the backtest sorts them itself."""
    vols = {"2025-06-16": 0.20, "2025-06-17": 0.25, "2025-06-18": 0.30}
    for date, vol in reversed(list(vols.items())):
        df = synthetic_snapshot(BSMModel({"sigma": vol}), snapshot_date=date, q=0.015)
        df.to_parquet(tmp_path / f"TEST_{date}.parquet")
    monkeypatch.setattr(market_data_manager, "MARKET_SNAPSHOT_DIR", tmp_path)
    monkeypatch.setattr(backtest_workflow, "BACKTEST_LOGS_DIR", tmp_path / "logs")
    return vols


def test_backtest_pairs_dates_in_ascending_order(corpus, no_network):
    """Regression: dates came newest-first and the loop calibrated on t+1 and
    evaluated on t (look-ahead). Now calibrate on t, evaluate on t+1."""
    bt = BacktestWorkflow("test", ALL_MODEL_CONFIGS["BSM"], settings=FAST)
    bt.run()
    assert [(r["Calibration Date"], r["Eval Date"]) for r in bt.results] == [
        ("2025-06-16", "2025-06-17"),
        ("2025-06-17", "2025-06-18"),
    ]
    assert bt.ticker == "TEST"
    for row, (calib_vol, eval_vol) in zip(bt.results, [(0.20, 0.25), (0.25, 0.30)]):
        assert row["Status"] == "Success"
        assert row["Params"]["sigma"] == pytest.approx(calib_vol, abs=1e-4)
        # a flat vol fitted on day t is off by exactly the day-t+1 vol change
        assert row["Out-of-Sample IV RMSE"] == pytest.approx(
            eval_vol - calib_vol, abs=2e-3
        )
        assert row["In-Sample IV RMSE"] < 1e-4
        assert row["BSM-ATM IV RMSE"] == pytest.approx(
            row["Out-of-Sample IV RMSE"], abs=2e-3
        )
        assert row["N Options"] > 0
    # the stale baseline needs two previous periods
    assert pd.isna(bt.results[0]["Stale-Params IV RMSE"])
    assert bt.results[1]["Stale-Params IV RMSE"] == pytest.approx(0.10, abs=2e-3)
    assert {b["bucket_type"] for b in bt.bucket_results} == {"maturity", "moneyness"}


def test_backtest_uses_each_days_market_parameters(corpus, no_network, monkeypatch):
    seen = []
    original = backtest_workflow.DailyWorkflow._prepare_for_evaluation

    def spy(self):
        original(self)
        seen.append(
            (
                self.market_data["snapshot_date"].iloc[0],
                self.rate.get_rate(),
                self.stock.dividend,
            )
        )

    monkeypatch.setattr(backtest_workflow.DailyWorkflow, "_prepare_for_evaluation", spy)
    BacktestWorkflow("TEST", ALL_MODEL_CONFIGS["BSM"], settings=FAST).run()
    assert all(
        r == pytest.approx(0.04) and q == pytest.approx(0.015) for _, r, q in seen
    )
    assert "2025-06-18" in {d for d, _, _ in seen}


def test_summary_and_save(corpus, no_network):
    bt = BacktestWorkflow("TEST", ALL_MODEL_CONFIGS["BSM"], settings=FAST)
    bt.run()
    summary = bt.summary()
    assert summary["n_periods"] == 2 and summary["n_failed"] == 0
    assert summary["mean_oos_iv_rmse"] == pytest.approx(0.05, abs=2e-3)
    assert 0 <= summary["share_beating_bsm_atm"] <= 1

    paths = bt.save_results()
    assert paths is not None
    csv_path, json_path = paths
    table = pd.read_csv(csv_path)
    assert list(table["Eval Date"]) == ["2025-06-17", "2025-06-18"]
    assert "Out-of-Sample RMSE" in table.columns
    payload = json.loads(json_path.read_text())
    assert payload["summary"]["model"] == "BSM"
    assert payload["settings"]["n_starts"] == 1
    assert payload["periods"][0]["Params"]["sigma"] == pytest.approx(0.20, abs=1e-4)
    assert (csv_path.parent / (csv_path.stem + "_buckets.csv")).exists()


def test_explicit_dates_and_too_few_dates(corpus, no_network, caplog):
    bt = BacktestWorkflow(
        "TEST",
        ALL_MODEL_CONFIGS["BSM"],
        settings=FAST,
        dates=["2025-06-18", "2025-06-16"],
    )
    bt.run()
    assert [(r["Calibration Date"], r["Eval Date"]) for r in bt.results] == [
        ("2025-06-16", "2025-06-18")
    ]

    short = BacktestWorkflow(
        "TEST", ALL_MODEL_CONFIGS["BSM"], settings=FAST, dates=["2025-06-16"]
    )
    short.run()
    assert short.results == [] and short.save_results() is None
    assert short.summary()["n_periods"] == 0


def test_failed_calibration_period_is_recorded(corpus, no_network):
    cfg = {**ALL_MODEL_CONFIGS["SABRJump"]}
    bt = BacktestWorkflow("TEST", cfg, settings=FAST)  # needs allow_mc
    bt.run()
    assert len(bt.results) == 2
    assert all(r["Status"] == "Failed" for r in bt.results)
    assert bt.summary()["n_failed"] == 2
