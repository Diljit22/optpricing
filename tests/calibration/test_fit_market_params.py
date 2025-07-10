import numpy as np
import pandas as pd
import pytest

from optpricing.calibration.fit_market_params import (
    find_atm_options,
    fit_rate_and_dividend,
)


@pytest.fixture
def sample_chain():
    """Provides sample call and put dataframes."""
    calls = pd.DataFrame(
        {
            "strike": [95, 100, 105, 100, 105],
            "maturity": [0.1, 0.1, 0.1, 0.2, 0.2],
            "marketPrice": [5.5, 2.0, 0.5, 3.0, 1.0],
        }
    )
    puts = pd.DataFrame(
        {
            "strike": [95, 100, 105, 100, 105],
            "maturity": [0.1, 0.1, 0.1, 0.2, 0.2],
            "marketPrice": [0.4, 1.8, 5.2, 2.5, 5.5],
        }
    )
    return calls, puts


def test_find_atm_options(sample_chain):
    """
    Tests correct identification of the closest ATM strike for each maturity.
    """
    calls, puts = sample_chain
    spot = 101.0

    atm_pairs = find_atm_options(calls, puts, spot)

    # Should find one pair for each maturity (0.1 and 0.2)
    assert len(atm_pairs) == 2

    # For T=0.1, strike 100 is closer to 101 than 95 or 105
    assert atm_pairs[atm_pairs["maturity"] == 0.1].iloc[0]["strike"] == 100

    # For T=0.2, strike 100 is closer to 101 than 105
    assert atm_pairs[atm_pairs["maturity"] == 0.2].iloc[0]["strike"] == 100


def test_fit_rate_and_dividend():
    """
    Tests that the function can recover known r and q from perfect parity prices.
    """
    spot = 100.0
    r_true, q_true = 0.05, 0.02

    # Create perfect ATM data
    atm_pairs = pd.DataFrame({"strike": [100, 100], "maturity": [0.5, 1.0]})

    # C - P = S*exp(-qT) - K*exp(-rT)
    parity_rhs = (spot * np.exp(-q_true * atm_pairs["maturity"])) - (
        atm_pairs["strike"] * np.exp(-r_true * atm_pairs["maturity"])
    )

    # We can set C=parity_rhs and P=0 for simplicity
    atm_pairs["marketPrice_call"] = parity_rhs
    atm_pairs["marketPrice_put"] = 0.0

    calls = atm_pairs[["strike", "maturity", "marketPrice_call"]].rename(
        columns={"marketPrice_call": "marketPrice"}
    )
    puts = atm_pairs[["strike", "maturity", "marketPrice_put"]].rename(
        columns={"marketPrice_put": "marketPrice"}
    )

    r_fit, q_fit = fit_rate_and_dividend(calls, puts, spot)

    assert r_fit == pytest.approx(r_true, abs=1e-2)
    assert q_fit == pytest.approx(q_true, abs=1e-2)
