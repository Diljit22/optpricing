import numpy as np
import pandas as pd
import pytest

from optpricing.atoms import Rate, Stock
from optpricing.calibration.vectorized_bsm_iv import BSMIVSolver, bsm_prices_and_vegas
from optpricing.models import BSMModel


@pytest.fixture
def setup():
    """Provides a standard setup for BSM IV tests."""
    options = pd.DataFrame(
        {
            "strike": [95, 100, 105],
            "maturity": [1.0, 1.0, 1.0],
            "optionType": ["call", "call", "call"],
        }
    )
    stock = Stock(spot=100)
    rate = Rate(rate=0.05)
    return options, stock, rate


def test_bsm_iv_solver(setup):
    """
    Tests that the solver can recover known volatilities from BSM prices.
    """
    options, stock, rate = setup

    target_vols = np.array([0.15, 0.20, 0.25])
    target_prices = np.zeros_like(target_vols)
    for i, vol in enumerate(target_vols):
        model = BSMModel(params={"sigma": vol})
        target_prices[i] = model.price_closed_form(
            spot=stock.spot,
            strike=options.loc[i, "strike"],
            r=rate.get_rate(options.loc[i, "maturity"]),
            q=stock.dividend,
            t=options.loc[i, "maturity"],
            call=True,
        )

    solver = BSMIVSolver()
    implied_vols = solver.solve(target_prices, options, stock, rate)

    np.testing.assert_allclose(implied_vols, target_vols, atol=1e-3)


def _random_chain(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    K = rng.uniform(50, 160, n)
    T = rng.choice([0.01, 0.05, 0.25, 1.0, 2.0], n)
    vols = rng.uniform(0.05, 1.5, n)
    is_call = rng.random(n) < 0.5
    options = pd.DataFrame(
        {"strike": K, "maturity": T, "optionType": np.where(is_call, "call", "put")}
    )
    prices, _ = bsm_prices_and_vegas(vols, 100.0, K, T, 0.04, 0.01, is_call)
    return options, vols, prices, is_call


def test_solver_handles_deep_itm_otm_and_short_maturities():
    """Every option with measurable time value is solved; the price is
    reproduced to the tolerance and the vol error is tiny. Options whose price
    equals the discounted intrinsic value within rounding are NaN rather than
    garbage (the 20-iteration plain Newton used to return whatever it landed on)."""
    options, vols, prices, is_call = _random_chain()
    stock, rate = Stock(spot=100.0, dividend=0.01), Rate(rate=0.04)
    iv = BSMIVSolver().solve(prices, options, stock, rate)

    K, T = options["strike"].to_numpy(), options["maturity"].to_numpy()
    fwd, kd = 100.0 * np.exp(-0.01 * T), K * np.exp(-0.04 * T)
    time_value = prices - np.maximum(np.where(is_call, fwd - kd, kd - fwd), 0.0)
    informative = time_value > 1e-4

    assert np.isfinite(iv[informative]).all()
    assert np.nanmax(np.abs(iv[informative] - vols[informative])) < 5e-4
    solved = np.isfinite(iv)
    repriced, _ = bsm_prices_and_vegas(
        iv[solved], 100.0, K[solved], T[solved], 0.04, 0.01, is_call[solved]
    )
    assert np.max(np.abs(repriced - prices[solved])) < 2e-6


def test_prices_outside_the_arbitrage_band_give_nan():
    options = pd.DataFrame(
        {
            "strike": [60.0, 100.0, 140.0],
            "maturity": [0.5] * 3,
            "optionType": ["call", "call", "put"],
        }
    )
    stock, rate = Stock(spot=100.0, dividend=0.0), Rate(rate=0.05)
    # below intrinsic, above the spot bound, exactly intrinsic (zero time value)
    targets = np.array([30.0, 101.0, 140.0 * np.exp(-0.025) - 100.0])
    iv = BSMIVSolver().solve(targets, options, stock, rate)
    assert np.isnan(iv).all()


def test_initial_guess_and_term_structure_rates():
    options, vols, prices, is_call = _random_chain(n=500, seed=3)
    stock = Stock(spot=100.0, dividend=0.01)
    flat = BSMIVSolver().solve(prices, options, stock, Rate(rate=0.04))
    warm = BSMIVSolver().solve(
        prices, options, stock, Rate(rate=0.04), initial_guess=vols * 1.1
    )
    K, T = options["strike"].to_numpy(), options["maturity"].to_numpy()
    fwd, kd = 100.0 * np.exp(-0.01 * T), K * np.exp(-0.04 * T)
    time_value = prices - np.maximum(np.where(is_call, fwd - kd, kd - fwd), 0.0)
    # Where the price pins down the vol, the start point does not matter.
    mask = np.isfinite(flat) & np.isfinite(warm) & (time_value > 1e-4)
    np.testing.assert_allclose(flat[mask], warm[mask], atol=1e-4)
    # a callable rate curve is evaluated per maturity
    curve = BSMIVSolver().solve(prices, options, stock, Rate(rate=lambda t: 0.04))
    np.testing.assert_allclose(curve[mask], flat[mask], atol=1e-6)
