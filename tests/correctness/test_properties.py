"""
Property-based tests (hypothesis): model-free identities that must hold for
every valid parameter set, not just the hand-picked ones elsewhere.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from optpricing import (
    BatesModel,
    BSMModel,
    CGMYModel,
    HestonModel,
    HyperbolicModel,
    KouModel,
    MertonJumpModel,
    NIGModel,
    Option,
    OptionType,
    Rate,
    Stock,
    VarianceGammaModel,
)
from optpricing.calibration.vectorized_pricer import ChainLayout, price_chain
from optpricing.parity import ParityModel
from optpricing.techniques import ClosedFormTechnique


def _f(lo, hi):
    return st.floats(lo, hi, allow_nan=False, allow_infinity=False)


_JUMPS = {"lambda": _f(0.0, 2.0), "mu_j": _f(-0.3, 0.1), "sigma_j": _f(0.01, 0.4)}
_SV = {
    "v0": _f(0.005, 0.5),
    "kappa": _f(0.2, 5.0),
    "theta": _f(0.005, 0.3),
    "rho": _f(-0.95, 0.5),
    "vol_of_vol": _f(0.05, 1.0),
}
_PARAMS = {
    "BSM": (BSMModel, {"sigma": _f(0.05, 1.0)}),
    "Heston": (HestonModel, _SV),
    "Bates": (BatesModel, {**_SV, **_JUMPS}),
    "Merton": (
        MertonJumpModel,
        {"sigma": _f(0.05, 0.6), **_JUMPS, "max_sum_terms": st.just(100)},
    ),
    "Kou": (
        KouModel,
        {
            "sigma": _f(0.05, 0.5),
            "lambda": _f(0.0, 3.0),
            "p_up": _f(0.05, 0.95),
            "eta1": _f(1.5, 30.0),
            "eta2": _f(1.0, 30.0),
        },
    ),
    "VG": (
        VarianceGammaModel,
        {"sigma": _f(0.05, 0.5), "nu": _f(0.02, 0.8), "theta": _f(-0.4, 0.2)},
    ),
    "NIG": (
        NIGModel,
        {"alpha": _f(2.0, 40.0), "beta": _f(-30.0, 30.0), "delta": _f(0.05, 2.0)},
    ),
    "CGMY": (
        CGMYModel,
        {
            "C": _f(0.005, 1.0),
            "G": _f(1.0, 20.0),
            "M": _f(1.5, 20.0),
            "Y": _f(0.0, 1.8),
        },
    ),
    "Hyperbolic": (
        HyperbolicModel,
        {
            "alpha": _f(2.0, 40.0),
            "beta": _f(-30.0, 30.0),
            "delta": _f(0.05, 2.0),
            "mu": st.just(0.0),
        },
    ),
}


def _valid_model(data, name):
    model_class, strategies = _PARAMS[name]
    params = {key: data.draw(s, label=key) for key, s in strategies.items()}
    try:
        model = model_class(params=params)
    except ValueError:  # outside the model's risk-neutral parameter region
        assume(False)
    kwargs = {"v0": params["v0"]} if "v0" in params else {}
    return model, kwargs


@pytest.mark.parametrize("name", list(_PARAMS))
@given(data=st.data(), T=_f(0.05, 5.0), r=_f(-0.01, 0.1), q=_f(0.0, 0.06))
def test_characteristic_function_is_a_normalised_martingale(name, data, T, r, q):
    """phi(0) = 1, phi(-i) = S e^{(r-q)T} and |phi(u)| <= 1 on the real line."""
    model, kwargs = _valid_model(data, name)
    S = 100.0
    phi = model.cf(t=T, spot=S, r=r, q=q, **kwargs)
    assert abs(phi(0.0) - 1.0) < 1e-10
    forward = S * math.exp((r - q) * T)
    assert np.real(phi(-1j)) == pytest.approx(forward, rel=1e-8)
    u = np.linspace(0.1, 200.0, 400)
    assert np.all(np.abs(phi(u)) <= 1.0 + 1e-9)


_STRIKES = np.linspace(70.0, 140.0, 15)


@pytest.mark.parametrize("name", list(_PARAMS))
@given(data=st.data(), T=_f(0.02, 3.0), r=_f(0.0, 0.08), q=_f(0.0, 0.05))
def test_prices_are_arbitrage_free_in_strike(name, data, T, r, q):
    """
    Merton (1973) bounds, calls decreasing and puts increasing in K with slopes
    within [-e^{-rT}, 0] and [0, e^{-rT}], and convexity (butterflies >= 0) —
    model-free consequences of a non-negative risk-neutral density.
    """
    model, kwargs = _valid_model(data, name)
    S, n = 100.0, _STRIKES.size
    chain = pd.DataFrame(
        {
            "strike": np.concatenate([_STRIKES, _STRIKES]),
            "maturity": T,
            "optionType": ["call"] * n + ["put"] * n,
        }
    )
    layout = ChainLayout(chain)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        prices = price_chain(
            layout, Stock(spot=S, dividend=q), model, Rate(rate=r), **kwargs
        )
    # Near-degenerate laws (e.g. CGMY with C T ~ 1e-4) exhaust the COS terms and
    # say so; the properties are claimed for converged prices.
    assume(not any("COS series truncated" in str(w.message) for w in caught))
    calls, puts = prices[:n], prices[n:]
    tol = 1e-6
    bounds = ParityModel(params={})
    for K, c, p in zip(_STRIKES, calls, puts):
        lo, hi = bounds.price_bounds(
            spot=S, strike=K, r=r, q=q, t=T, call=True, option_price=c
        )
        assert lo - tol <= c <= hi + tol
        lo, hi = bounds.price_bounds(
            spot=S, strike=K, r=r, q=q, t=T, call=False, option_price=p
        )
        assert lo - tol <= p <= hi + tol
    dK, df = _STRIKES[1] - _STRIKES[0], math.exp(-r * T)
    call_slope = np.diff(calls) / dK
    assert np.all(call_slope <= tol) and np.all(call_slope >= -df - tol)
    assert np.all(np.diff(call_slope) >= -tol / dK)
    put_slope = np.diff(puts) / dK
    assert np.all(put_slope >= -tol) and np.all(put_slope <= df + tol)


_market = {
    "S": _f(20.0, 500.0),
    "moneyness": _f(0.5, 2.0),
    "T": _f(0.02, 5.0),
    "r": _f(-0.01, 0.1),
    "q": _f(0.0, 0.06),
    "sigma": _f(0.05, 1.5),
}


def _bsm_price(S, K, T, r, q, sigma, call):
    return BSMModel(params={"sigma": sigma}).price_closed_form(
        spot=S, strike=K, r=r, q=q, t=T, call=call
    )


@given(**_market)
def test_bsm_put_call_parity_and_monotonicity(S, moneyness, T, r, q, sigma):
    K = S * moneyness
    call = _bsm_price(S, K, T, r, q, sigma, True)
    put = _bsm_price(S, K, T, r, q, sigma, False)
    assert call - put == pytest.approx(
        S * math.exp(-q * T) - K * math.exp(-r * T), abs=1e-9 * S
    )
    assert _bsm_price(S, K, T, r, q, sigma * 1.01, True) >= call - 1e-12 * S
    assert _bsm_price(S * 1.01, K, T, r, q, sigma, True) >= call - 1e-12 * S
    assert _bsm_price(S * 1.01, K, T, r, q, sigma, False) <= put + 1e-12 * S


@given(**_market, call=st.booleans())
def test_scalar_implied_volatility_round_trip(S, moneyness, T, r, q, sigma, call):
    """Wherever the price pins sigma down (vega not negligible), the IV solver
    recovers it; elsewhere it must at least reproduce the price."""
    K = S * moneyness
    price = _bsm_price(S, K, T, r, q, sigma, call)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    vega = S * math.exp(-q * T) * math.sqrt(T) * math.exp(-0.5 * d1 * d1)
    vega /= math.sqrt(2.0 * math.pi)
    assume(vega > 1e-3 * S)  # a 1e-6 price tolerance then means < 1e-3 in sigma
    option = Option(
        strike=K, maturity=T, option_type=OptionType.CALL if call else OptionType.PUT
    )
    iv = ClosedFormTechnique().implied_volatility(
        option,
        Stock(spot=S, dividend=q),
        BSMModel(params={"sigma": 0.3}),
        Rate(rate=r),
        target_price=price,
    )
    assert _bsm_price(S, K, T, r, q, iv, call) == pytest.approx(price, abs=1e-6 * S)
    assert iv == pytest.approx(sigma, abs=1e-3)
