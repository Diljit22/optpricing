import numpy as np
import pandas as pd
import pytest

from optpricing.calibration.fit_market_params import (
    find_atm_options,
    fit_rate_and_dividend,
    implied_dividend_yield,
    implied_forward_curve,
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


def european_parity_chain(spot, r, q, maturities, strikes, *, put_eep=0.0):
    """Calls/puts whose mids satisfy parity; `put_eep` adds an early-exercise
    premium growing with the put's moneyness (what American puts do)."""
    rows_c, rows_p = [], []
    for T in maturities:
        F = spot * np.exp((r - q) * T)
        df = np.exp(-r * T)
        for K in strikes:
            call = df * max(F - K, 0.0) + 2.0
            put = call - df * (F - K) + put_eep * max(K - F, 0.0)
            rows_c.append({"strike": K, "maturity": T, "marketPrice": call})
            rows_p.append({"strike": K, "maturity": T, "marketPrice": put})
    return pd.DataFrame(rows_c), pd.DataFrame(rows_p)


def test_find_atm_options(sample_chain):
    calls, puts = sample_chain
    atm_pairs = find_atm_options(calls, puts, 101.0)
    assert len(atm_pairs) == 2
    assert atm_pairs[atm_pairs["maturity"] == 0.1].iloc[0]["strike"] == 100
    assert atm_pairs[atm_pairs["maturity"] == 0.2].iloc[0]["strike"] == 100


def test_fit_rate_and_dividend_legacy_atm_fit():
    """Recovers r and q from perfect single-strike parity data (legacy path)."""
    spot, r_true, q_true = 100.0, 0.05, 0.02
    atm_pairs = pd.DataFrame({"strike": [100, 100], "maturity": [0.5, 1.0]})
    parity_rhs = (spot * np.exp(-q_true * atm_pairs["maturity"])) - (
        atm_pairs["strike"] * np.exp(-r_true * atm_pairs["maturity"])
    )
    calls = atm_pairs.assign(marketPrice=parity_rhs)[
        ["strike", "maturity", "marketPrice"]
    ]
    puts = atm_pairs.assign(marketPrice=0.0)[["strike", "maturity", "marketPrice"]]

    r_fit, q_fit = fit_rate_and_dividend(calls, puts, spot)
    assert r_fit == pytest.approx(r_true, abs=1e-2)
    assert q_fit == pytest.approx(q_true, abs=1e-2)


def test_implied_forward_curve_recovers_forward_and_dividend():
    spot, r, q = 600.0, 0.043, 0.012
    maturities = (0.25, 0.5, 1.0)
    calls, puts = european_parity_chain(
        spot, r, q, maturities, np.arange(560, 641, 5.0)
    )
    curve = implied_forward_curve(calls, puts, spot, r)
    assert list(curve["maturity"]) == list(maturities)
    for _, row in curve.iterrows():
        assert row["forward"] == pytest.approx(
            spot * np.exp((r - q) * row["maturity"]), rel=1e-10
        )
        assert row["implied_dividend"] == pytest.approx(q, abs=1e-10)
    assert implied_dividend_yield(calls, puts, spot, r) == pytest.approx(q, abs=1e-10)


def test_implied_dividend_is_robust_to_american_put_premium():
    """Deep ITM American puts bias the strike slope of C-P towards -1 (seen on
    real SPY chains, where a regression implied r ~ 0 and q ~ -3 %). Using only
    near-the-money pairs with a known rate keeps the implied yield right."""
    spot, r, q = 600.0, 0.043, 0.012
    calls, puts = european_parity_chain(
        spot, r, q, (1.0,), np.arange(540, 661, 5.0), put_eep=0.04
    )
    q_hat = implied_dividend_yield(calls, puts, spot, r)
    assert q_hat == pytest.approx(q, abs=2e-3)

    # The naive full-window regression is biased by >100 bp on the same data.
    merged = calls.merge(puts, on=["strike", "maturity"], suffixes=("_c", "_p"))
    slope, _ = np.polyfit(
        merged["strike"], merged["marketPrice_c"] - merged["marketPrice_p"], 1
    )
    assert -np.log(-slope) < r - 0.01


def test_fit_rate_and_dividend_with_fixed_rate_uses_forward_curve():
    spot, r, q = 100.0, 0.03, 0.02
    calls, puts = european_parity_chain(spot, r, q, (0.5, 1.0), np.arange(90, 111, 2.0))
    r_fit, q_fit = fit_rate_and_dividend(calls, puts, spot, r_fixed=r)
    assert r_fit == r
    assert q_fit == pytest.approx(q, abs=1e-8)
    assert fit_rate_and_dividend(calls, puts, spot, r_fixed=r, q_fixed=0.0) == (r, 0.0)


def test_implied_dividend_yield_handles_empty_and_implausible():
    empty = pd.DataFrame(columns=["strike", "maturity", "marketPrice"])
    assert implied_dividend_yield(empty, empty, 100.0, 0.04) == 0.0
    # Pairs implying a 60 % yield are rejected in favour of the default.
    calls, puts = european_parity_chain(
        100.0, 0.04, 0.60, (1.0,), np.arange(60, 141, 5.0)
    )
    assert implied_dividend_yield(calls, puts, 100.0, 0.04) == 0.0


def test_implied_forward_curve_widens_window_to_min_pairs():
    """Three pairs are used even when none is inside the 3 % window."""
    calls, puts = european_parity_chain(
        100.0, 0.02, 0.0, (0.5,), [80.0, 120.0, 125.0, 130.0]
    )
    curve = implied_forward_curve(calls, puts, 100.0, 0.02)
    assert curve["n_pairs"].iloc[0] == 3
    assert curve["forward"].iloc[0] == pytest.approx(100.0 * np.exp(0.01))
