from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from optpricing.atoms import Rate, Stock
from optpricing.calibration.vectorized_pricer import (
    price_options_vectorized,
    supports_vectorized_pricing,
)
from optpricing.models import (
    BSMModel,
    CEVModel,
    DupireLocalVolModel,
    HestonModel,
    MertonJumpModel,
    SABRJumpModel,
    SABRModel,
    VarianceGammaModel,
)


def bsm_closed_form(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    is_call: bool,
) -> float:
    """Standard BSM closed-form solution for testing."""
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if is_call:
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


@pytest.fixture
def setup_vectorized_test():
    """Provides a standard setup for the vectorized pricer test."""
    stock = Stock(spot=100.0, dividend=0.01)
    rate = Rate(rate=0.05)
    model = BSMModel(params={"sigma": 0.20})

    # Create a sample DataFrame with a mix of options
    options_df = pd.DataFrame(
        {
            "strike": [95.0, 100.0, 105.0],
            "maturity": [1.0, 1.0, 1.0],
            "optionType": ["call", "call", "put"],
        }
    )
    return options_df, stock, model, rate


@pytest.fixture
def chain():
    strikes = np.linspace(80.0, 120.0, 21)
    return pd.DataFrame(
        {
            "strike": np.tile(strikes, 3),
            "maturity": np.repeat([0.1, 0.5, 1.0], len(strikes)),
            "optionType": np.tile(
                np.where(np.arange(len(strikes)) % 2 == 0, "call", "put"), 3
            ),
        }
    )


def test_price_options_vectorized(setup_vectorized_test):
    """
    Tests that the vectorized pricer matches the closed-form BSM solution.
    """
    options_df, stock, model, rate = setup_vectorized_test

    # Calculate prices using the vectorized function
    vectorized_prices = price_options_vectorized(options_df, stock, model, rate)

    # Calculate expected prices using the closed-form solution one-by-one
    expected_prices = np.array(
        [
            bsm_closed_form(
                S=stock.spot,
                K=row.strike,
                T=row.maturity,
                r=rate.get_rate(row.maturity),
                q=stock.dividend,
                sigma=model.params["sigma"],
                is_call=(row.optionType == "call"),
            )
            for _, row in options_df.iterrows()
        ]
    )

    assert len(vectorized_prices) == len(expected_prices)
    np.testing.assert_allclose(vectorized_prices, expected_prices, rtol=1e-5)


def test_closed_form_route_agrees_with_characteristic_function_route(chain):
    """Merton has both; the two routes must give the same chain prices."""
    stock, rate = Stock(spot=100.0, dividend=0.01), Rate(rate=0.04)
    model = MertonJumpModel(params=MertonJumpModel.default_params)

    class CFOnly(MertonJumpModel):
        has_closed_form = False
        vectorized_closed_form = False

    via_closed_form = price_options_vectorized(chain, stock, model, rate)
    via_cf = price_options_vectorized(
        chain, stock, CFOnly(params=MertonJumpModel.default_params), rate
    )
    np.testing.assert_allclose(via_closed_form, via_cf, atol=1e-8)


@pytest.mark.parametrize(
    "model",
    [SABRModel(), CEVModel()],
    ids=["SABR", "CEV"],
)
def test_models_without_cf_but_with_closed_form_are_priced(chain, model):
    """Previously raised NotImplementedError (the pricer required a CF)."""
    stock, rate = Stock(spot=100.0, dividend=0.01), Rate(rate=0.04)
    prices = price_options_vectorized(chain, stock, model, rate)
    expected = [
        model.price_closed_form(
            spot=100.0,
            strike=row.strike,
            r=0.04,
            q=0.01,
            t=row.maturity,
            call=row.optionType == "call",
        )
        for row in chain.itertuples()
    ]
    np.testing.assert_allclose(prices, expected, rtol=1e-12)


def test_cf_models_and_pure_levy_are_supported():
    assert supports_vectorized_pricing(HestonModel())
    assert supports_vectorized_pricing(VarianceGammaModel())
    assert not supports_vectorized_pricing(SABRJumpModel())
    assert supports_vectorized_pricing(SABRJumpModel(), allow_mc=True)
    assert supports_vectorized_pricing(DupireLocalVolModel(), allow_mc=True)


def test_monte_carlo_fallback_is_opt_in_and_deterministic(chain):
    stock, rate = Stock(spot=100.0, dividend=0.01), Rate(rate=0.04)
    model = SABRJumpModel()
    with pytest.raises(NotImplementedError, match="allow_mc"):
        price_options_vectorized(chain, stock, model, rate)

    kwargs = {"allow_mc": True, "mc_paths": 4000, "mc_steps": 10, "mc_seed": 7}
    a = price_options_vectorized(chain, stock, model, rate, **kwargs)
    b = price_options_vectorized(chain, stock, model, rate, **kwargs)
    np.testing.assert_array_equal(a, b)  # common random numbers
    assert np.all(a >= 0)
    # Deep ITM call (K=80, T=0.1) is worth about its intrinsic value.
    assert a[0] == pytest.approx(20.0 * np.exp(-0.04 * 0.1), abs=1.5)


def test_monte_carlo_fallback_matches_closed_form_for_bsm(chain):
    """The MC route prices the whole strike grid from one set of terminal spots."""
    stock, rate = Stock(spot=100.0, dividend=0.01), Rate(rate=0.04)

    class MCOnlyBSM(BSMModel):
        supports_cf = False
        has_closed_form = False
        vectorized_closed_form = False

    model = MCOnlyBSM(params={"sigma": 0.2})
    exact = price_options_vectorized(chain, stock, BSMModel({"sigma": 0.2}), rate)
    mc = price_options_vectorized(
        chain, stock, model, rate, allow_mc=True, mc_paths=200_000, mc_steps=5
    )
    np.testing.assert_allclose(mc, exact, atol=0.1)


def test_empty_chain_returns_empty_array():
    empty = pd.DataFrame(columns=["strike", "maturity", "optionType"])
    out = price_options_vectorized(empty, Stock(100.0), BSMModel(), Rate(0.01))
    assert out.shape == (0,)
