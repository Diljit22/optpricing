import numpy as np
import pandas as pd
import pytest

from optpricing.atoms import Rate, Stock
from optpricing.calibration import CalibrationResult, Calibrator, filter_options
from optpricing.models import (
    BSMModel,
    HestonModel,
    MertonJumpModel,
    SABRJumpModel,
    SABRModel,
)

STOCK, RATE = Stock(spot=100.0, dividend=0.01), Rate(rate=0.04)
HESTON_TRUE = {"v0": 0.05, "kappa": 1.5, "theta": 0.06, "rho": -0.6, "vol_of_vol": 0.5}
HESTON_BOUNDS = {
    "v0": (0.001, 0.5),
    "kappa": (0.1, 10.0),
    "theta": (0.01, 0.5),
    "rho": (-0.99, 0.99),
    "vol_of_vol": (0.1, 1.5),
}


def test_recovers_bsm_volatility_in_iv_space(synthetic_snapshot):
    chain = synthetic_snapshot(BSMModel(params={"sigma": 0.27}))
    data = filter_options(chain, STOCK, RATE)
    cal = Calibrator(BSMModel(), data, STOCK, RATE)
    params = cal.fit({"sigma": 0.2}, {"sigma": (0.01, 2.0)})
    assert params["sigma"] == pytest.approx(0.27, abs=1e-4)
    res = cal.result
    assert isinstance(res, CalibrationResult)
    assert res.rmse_iv < 1e-4 and res.rmse_price < 1e-3
    assert res.n_options == len(data) and res.success


@pytest.mark.parametrize("objective", ["iv", "price"])
def test_recovers_heston_parameters_from_its_own_prices(synthetic_snapshot, objective):
    """Noise-free Heston chain: the multi-start IV/price fit lands on the
    generating smile (parameters recovered to a few percent; Heston has the
    usual kappa/vol-of-vol trade-off, so the smile fit is the real criterion)."""
    truth = HestonModel(params=HESTON_TRUE)
    chain = synthetic_snapshot(truth, maturities=(0.1, 0.25, 0.5, 1.0, 2.0))
    data = filter_options(chain, STOCK, RATE)
    cal = Calibrator(HestonModel(), data, STOCK, RATE, objective=objective)
    params = cal.fit(
        {"v0": 0.04, "kappa": 2.0, "theta": 0.04, "rho": -0.7, "vol_of_vol": 0.3},
        HESTON_BOUNDS,
        n_starts=4,
    )
    assert cal.result.rmse_iv < 2e-3
    assert params["v0"] == pytest.approx(HESTON_TRUE["v0"], rel=0.05)
    assert params["rho"] == pytest.approx(HESTON_TRUE["rho"], abs=0.08)


def test_merton_fit_with_frozen_and_warm_start(synthetic_snapshot):
    true = {"sigma": 0.15, "lambda": 0.8, "mu_j": -0.2, "sigma_j": 0.2}
    chain = synthetic_snapshot(MertonJumpModel(params={**true, "max_sum_terms": 100}))
    data = filter_options(chain, STOCK, RATE)
    cal = Calibrator(
        MertonJumpModel(params=MertonJumpModel.default_params), data, STOCK, RATE
    )
    params = cal.fit(
        {"sigma": 0.2, "lambda": 0.5, "mu_j": -0.1, "sigma_j": 0.15},
        {
            "sigma": (0.01, 1.0),
            "lambda": (0.0, 5.0),
            "mu_j": (-1.0, 1.0),
            "sigma_j": (0.01, 1.0),
        },
        frozen_params={"lambda": 0.8},
        warm_start={"sigma": 0.15, "mu_j": -0.2, "sigma_j": 0.2},
        n_starts=2,
    )
    assert params["lambda"] == 0.8  # frozen value passed through untouched
    assert params["sigma"] == pytest.approx(0.15, abs=0.01)
    assert cal.result.rmse_iv < 1e-3
    assert {s["kind"] for s in cal.result.starts} <= {"guess", "warm", "lhs"}
    assert cal.result.n_starts == 2


def test_nothing_to_fit_returns_frozen(bsm_snapshot):
    cal = Calibrator(BSMModel(), bsm_snapshot, STOCK, RATE)
    assert cal.fit({"sigma": 0.2}, {"sigma": (0.01, 1)}, {"sigma": 0.3}) == {
        "sigma": 0.3
    }
    assert cal.result.message == "nothing to fit"


def test_invalid_parameters_are_penalised_not_raised(bsm_snapshot):
    cal = Calibrator(SABRModel(), bsm_snapshot, STOCK, RATE)
    # rho outside (-1, 1) makes the model constructor raise ValueError
    bad = cal.loss({"alpha": 0.2, "beta": 1.0, "rho": 1.5, "nu": 0.5})
    good = cal.loss({"alpha": 0.2, "beta": 1.0, "rho": -0.3, "nu": 0.5})
    assert bad == cal._invalid_loss > good
    # legacy entry point still works
    legacy = cal._objective_function(
        [0.2, 1.0], ["alpha", "beta"], {"rho": -0.3, "nu": 0.5}
    )
    assert legacy == pytest.approx(
        cal.loss({"alpha": 0.2, "beta": 1.0, "rho": -0.3, "nu": 0.5})
    )


def test_weights_and_objectives(bsm_snapshot):
    data = filter_options(bsm_snapshot, STOCK, RATE)
    spread = Calibrator(BSMModel(), data, STOCK, RATE, weighting="spread")
    none = Calibrator(BSMModel(), data, STOCK, RATE, weighting="none")
    vega = Calibrator(
        BSMModel(), data, STOCK, RATE, objective="price", weighting="vega"
    )
    for cal in (spread, none, vega):
        assert cal._weights.shape == (len(data),)
        assert cal._weights.mean() == pytest.approx(1.0)
    assert np.all(none._weights == 1.0)
    assert spread._weights.std() > 0  # wider (absolute) spreads weigh less
    # IV objective of the generating sigma is ~0; a wrong sigma is not
    assert spread.loss({"sigma": 0.23}) < 1e-8 < spread.loss({"sigma": 0.3})
    assert vega.loss({"sigma": 0.23}) < 1e-6 < vega.loss({"sigma": 0.3})
    with pytest.raises(ValueError):
        Calibrator(BSMModel(), data, STOCK, RATE, objective="foo")


def test_monte_carlo_only_models_need_allow_mc(bsm_snapshot):
    with pytest.raises(NotImplementedError, match="allow_mc"):
        Calibrator(SABRJumpModel(), bsm_snapshot, STOCK, RATE)
    cal = Calibrator(
        SABRJumpModel(),
        bsm_snapshot.head(20),
        STOCK,
        RATE,
        allow_mc=True,
        mc_paths=2000,
        mc_steps=5,
    )
    assert np.isfinite(cal.loss(dict(SABRJumpModel.default_params)))


def test_empty_calibration_set_raises(bsm_snapshot):
    cal = Calibrator(BSMModel(), bsm_snapshot.iloc[0:0], STOCK, RATE)
    with pytest.raises(ValueError, match="No options"):
        cal.fit({"sigma": 0.2}, {"sigma": (0.01, 1.0)})


def test_time_budget_limits_local_starts(bsm_snapshot):
    data = filter_options(bsm_snapshot, STOCK, RATE)
    cal = Calibrator(BSMModel(), data, STOCK, RATE)
    cal.fit({"sigma": 0.5}, {"sigma": (0.01, 2.0)}, n_starts=6, time_budget=0.0)
    assert len(cal.result.starts) == 1  # only the best start was optimised
    assert cal.result.params["sigma"] == pytest.approx(0.23, abs=1e-3)


def test_feller_penalty_discourages_violations(bsm_snapshot):
    data = filter_options(bsm_snapshot, STOCK, RATE)
    plain = Calibrator(HestonModel(), data, STOCK, RATE)
    penalised = Calibrator(HestonModel(), data, STOCK, RATE, feller_penalty=10.0)
    violating = {
        "v0": 0.05,
        "kappa": 0.5,
        "theta": 0.04,
        "rho": -0.5,
        "vol_of_vol": 1.0,
    }
    assert penalised.loss(violating) > plain.loss(violating)
    ok = {"v0": 0.05, "kappa": 3.0, "theta": 0.04, "rho": -0.5, "vol_of_vol": 0.3}
    assert penalised.loss(ok) == pytest.approx(plain.loss(ok))


def test_price_objective_matches_legacy_sse_up_to_normalisation():
    """Objective = weighted mean squared price error (the old class summed)."""
    market = pd.DataFrame(
        {
            "strike": [100.0],
            "maturity": [1.0],
            "optionType": ["call"],
            "marketPrice": [10.0],
        }
    )
    cal = Calibrator(
        BSMModel(),
        market,
        Stock(spot=100),
        Rate(rate=0.05),
        objective="price",
        weighting="none",
    )
    model_price = BSMModel({"sigma": 0.2}).price_closed_form(
        spot=100, strike=100, r=0.05, q=0.0, t=1.0
    )
    assert cal.loss({"sigma": 0.2}) == pytest.approx((model_price - 10.0) ** 2)
