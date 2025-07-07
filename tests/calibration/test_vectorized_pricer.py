from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from optpricing.atoms import Rate, Stock
from optpricing.calibration.vectorized_pricer import price_options_vectorized
from optpricing.models import BSMModel


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
