"""
Model-independent invariants: martingale condition of every characteristic
function, Levy-process consistency, put-call parity, limiting cases,
no-arbitrage bounds, finite-difference Greeks and implied-volatility round trips.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from optpricing import (
    BatesModel,
    BSMModel,
    CGMYModel,
    HestonModel,
    HyperbolicModel,
    KouModel,
    MertonJumpModel,
    NIGModel,
    VarianceGammaModel,
)
from optpricing.techniques import (
    ClosedFormTechnique,
    FFTTechnique,
    IntegrationTechnique,
)
from optpricing.techniques.base import GreekMixin

from ._helpers import make_option

CF_MODEL_NAMES = [
    "BSM",
    "Heston",
    "Merton",
    "Bates",
    "Kou",
    "VG",
    "NIG",
    "CGMY",
    "Hyperbolic",
]
LEVY_MODEL_NAMES = ["VG", "NIG", "CGMY", "Hyperbolic"]


@pytest.mark.parametrize("name", CF_MODEL_NAMES)
@pytest.mark.parametrize("T", [0.25, 1.3, 5.0])
def test_characteristic_function_is_a_martingale(cf_models, name, T):
    """phi(-i) = E[S_T] must equal the forward S e^{(r-q)T} for every model."""
    model, kwargs = cf_models[name]
    S, r, q = 100.0, 0.05, 0.01
    phi = model.cf(t=T, spot=S, r=r, q=q, **kwargs)
    value = phi(-1j)
    assert np.real(value) == pytest.approx(S * math.exp((r - q) * T), rel=1e-9)
    assert abs(np.imag(value)) < 1e-7


@pytest.mark.parametrize("name", LEVY_MODEL_NAMES)
def test_levy_increments_are_infinitely_divisible(cf_models, name):
    """For a Levy process phi_t = phi_1^t; integer powers avoid branch issues."""
    model, _ = cf_models[name]
    u = np.array([0.3, 1.0, 2.5, 7.0, 15.0])
    np.testing.assert_allclose(
        model.raw_cf(t=2.0)(u), model.raw_cf(t=1.0)(u) ** 2, rtol=1e-10, atol=1e-12
    )
    np.testing.assert_allclose(
        model.raw_cf(t=0.5)(u) ** 2, model.raw_cf(t=1.0)(u), rtol=1e-10, atol=1e-12
    )


@pytest.mark.parametrize("name", CF_MODEL_NAMES)
def test_fft_agrees_with_quadrature(cf_models, name, stock, rate, atm_call):
    model, kwargs = cf_models[name]
    p_int = IntegrationTechnique().price(atm_call, stock, model, rate, **kwargs).price
    p_fft = FFTTechnique().price(atm_call, stock, model, rate, **kwargs).price
    assert p_fft == pytest.approx(p_int, abs=5e-3)


@pytest.mark.parametrize("name", CF_MODEL_NAMES)
@pytest.mark.parametrize("technique", [IntegrationTechnique, FFTTechnique])
def test_put_call_parity(cf_models, name, technique, stock, rate):
    model, kwargs = cf_models[name]
    T, K = 1.0, 105.0
    call = (
        technique().price(make_option(K, T, True), stock, model, rate, **kwargs).price
    )
    put = (
        technique().price(make_option(K, T, False), stock, model, rate, **kwargs).price
    )
    forward_diff = stock.spot * math.exp(-stock.dividend * T) - K * math.exp(
        -rate.rate * T
    )
    assert call - put == pytest.approx(forward_diff, abs=1e-6)


@pytest.mark.parametrize("name", CF_MODEL_NAMES)
def test_no_arbitrage_bounds(cf_models, name, stock, rate):
    model, kwargs = cf_models[name]
    for K in (60.0, 100.0, 140.0):
        T = 1.0
        call = (
            IntegrationTechnique()
            .price(make_option(K, T), stock, model, rate, **kwargs)
            .price
        )
        lower = max(
            stock.spot * math.exp(-stock.dividend * T) - K * math.exp(-rate.rate * T),
            0.0,
        )
        assert lower - 1e-6 <= call <= stock.spot * math.exp(-stock.dividend * T) + 1e-6


def test_heston_reduces_to_bsm_as_vol_of_vol_vanishes(stock, rate, atm_call):
    bsm = (
        ClosedFormTechnique()
        .price(atm_call, stock, BSMModel(params={"sigma": 0.2}), rate)
        .price
    )
    heston = HestonModel(
        params={
            "v0": 0.04,
            "kappa": 2.0,
            "theta": 0.04,
            "rho": -0.7,
            "vol_of_vol": 1e-4,
        }
    )
    assert IntegrationTechnique().price(
        atm_call, stock, heston, rate, v0=0.04
    ).price == pytest.approx(bsm, abs=1e-4)


def test_merton_and_kou_reduce_to_bsm_as_intensity_vanishes(stock, rate, atm_call):
    bsm = (
        ClosedFormTechnique()
        .price(atm_call, stock, BSMModel(params={"sigma": 0.2}), rate)
        .price
    )
    merton = MertonJumpModel(
        params={
            "sigma": 0.2,
            "lambda": 1e-9,
            "mu_j": -0.1,
            "sigma_j": 0.15,
            "max_sum_terms": 50,
        }
    )
    kou = KouModel(
        params={"sigma": 0.2, "lambda": 1e-9, "p_up": 0.6, "eta1": 10.0, "eta2": 5.0}
    )
    assert ClosedFormTechnique().price(
        atm_call, stock, merton, rate
    ).price == pytest.approx(bsm, abs=1e-6)
    assert IntegrationTechnique().price(
        atm_call, stock, kou, rate
    ).price == pytest.approx(bsm, abs=1e-6)


def test_bates_reduces_to_heston_as_intensity_vanishes(
    stock, rate, atm_call, heston_base
):
    bates = BatesModel(
        params={**heston_base.params, "lambda": 1e-9, "mu_j": -0.1, "sigma_j": 0.15}
    )
    p_heston = (
        IntegrationTechnique().price(atm_call, stock, heston_base, rate, v0=0.04).price
    )
    p_bates = IntegrationTechnique().price(atm_call, stock, bates, rate, v0=0.04).price
    assert p_bates == pytest.approx(p_heston, abs=1e-6)


def test_variance_gamma_reduces_to_bsm_as_nu_vanishes(stock, rate, atm_call):
    bsm = (
        ClosedFormTechnique()
        .price(atm_call, stock, BSMModel(params={"sigma": 0.2}), rate)
        .price
    )
    vg = VarianceGammaModel(params={"sigma": 0.2, "nu": 1e-4, "theta": 0.0})
    assert IntegrationTechnique().price(
        atm_call, stock, vg, rate
    ).price == pytest.approx(bsm, abs=1e-3)


@pytest.mark.parametrize(
    "greek, tol",
    [("delta", 1e-5), ("gamma", 1e-5), ("vega", 1e-4), ("theta", 1e-4), ("rho", 1e-4)],
)
def test_finite_difference_greeks_match_analytic_bsm(stock, rate, atm_call, greek, tol):
    """GreekMixin central differences (driven through the quadrature pricer)
    against the analytic BSM Greeks."""
    bsm = BSMModel(params={"sigma": 0.2})
    analytic = getattr(ClosedFormTechnique(), greek)(atm_call, stock, bsm, rate)
    numeric = getattr(GreekMixin, greek)(
        IntegrationTechnique(), atm_call, stock, bsm, rate
    )
    assert numeric == pytest.approx(analytic, abs=tol)


@pytest.mark.parametrize(
    "strike, maturity, sigma",
    [
        (100.0, 1.0, 0.2),
        (60.0, 0.1, 0.3),
        (150.0, 0.1, 0.3),
        (100.0, 0.01, 0.1),
        (100.0, 5.0, 0.8),
    ],
)
def test_implied_volatility_round_trip(stock, rate, strike, maturity, sigma):
    cf = ClosedFormTechnique()
    option = make_option(strike, maturity)
    target = cf.price(option, stock, BSMModel(params={"sigma": sigma}), rate).price
    iv = cf.implied_volatility(
        option, stock, BSMModel(params={"sigma": 0.3}), rate, target_price=target
    )
    assert iv == pytest.approx(sigma, abs=1e-4)


@pytest.mark.parametrize(
    "model_cls, params, match",
    [
        (
            KouModel,
            {"sigma": 0.15, "lambda": 1.0, "p_up": 0.6, "eta1": 0.9, "eta2": 5.0},
            "eta1",
        ),
        (VarianceGammaModel, {"sigma": 0.5, "nu": 2.0, "theta": 0.4}, "theta\\*nu"),
        (NIGModel, {"alpha": 5.0, "beta": 4.5, "delta": 0.5}, "beta \\+ 1"),
        (
            HyperbolicModel,
            {"alpha": 5.0, "beta": 4.5, "delta": 0.5, "mu": 0.0},
            "beta \\+ 1",
        ),
        (CGMYModel, {"C": 0.02, "G": 5.0, "M": 1.0, "Y": 1.2}, "M must exceed 1"),
    ],
    ids=[
        "kou_eta1",
        "vg_exponential_moment",
        "nig_exponential_moment",
        "hyperbolic_exponential_moment",
        "cgmy_M",
    ],
)
def test_parameters_without_risk_neutral_measure_are_rejected(model_cls, params, match):
    with pytest.raises(ValueError, match=match):
        model_cls(params=params)


def test_heston_and_bates_default_v0_from_model_params(
    stock, rate, atm_call, heston_base
):
    """Regression: pricing raised TypeError unless v0 was re-passed as a kwarg
    even though it is already a model parameter."""
    with_kwarg = (
        IntegrationTechnique().price(atm_call, stock, heston_base, rate, v0=0.04).price
    )
    without_kwarg = (
        IntegrationTechnique().price(atm_call, stock, heston_base, rate).price
    )
    assert without_kwarg == pytest.approx(with_kwarg, abs=1e-12)

    no_v0 = HestonModel(
        params={"kappa": 2.0, "theta": 0.05, "rho": -0.7, "vol_of_vol": 0.5}
    )
    with pytest.raises(ValueError, match="v0"):
        IntegrationTechnique().price(atm_call, stock, no_v0, rate)


def test_sabr_put_never_pays_more_than_strike(stock, rate):
    """Regression: the SABR kernels let the spot go negative after a large
    down-move (only the diffusion coefficient was floored), so puts could pay
    K - S > K. The origin is now absorbing; the put is bounded by K e^{-rT}."""
    from optpricing import SABRModel
    from optpricing.techniques import MonteCarloTechnique

    sabr = SABRModel(params={"alpha": 3.0, "beta": 0.3, "rho": -0.5, "nu": 1.5})
    put = make_option(100.0, 2.0, call=False)
    price = (
        MonteCarloTechnique(n_paths=100_000, n_steps=50, seed=9)
        .price(put, stock, sabr, rate, v0=3.0)
        .price
    )
    assert price <= 100.0 * math.exp(-rate.rate * 2.0) + 1e-9
