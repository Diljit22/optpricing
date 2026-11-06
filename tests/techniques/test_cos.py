import math
import warnings
from unittest.mock import MagicMock

import numpy as np
import pytest

from optpricing.atoms import Option, OptionType, Rate, Stock
from optpricing.models import (
    BSMModel,
    CGMYModel,
    HyperbolicModel,
    KouModel,
    MertonJumpModel,
    NIGModel,
    VarianceGammaModel,
)
from optpricing.techniques import COSTechnique
from optpricing.techniques.cos import cos_prices


@pytest.fixture
def setup():
    option = Option(strike=105, maturity=1.0, option_type=OptionType.CALL)
    stock = Stock(spot=100, dividend=0.01)
    model = BSMModel(params={"sigma": 0.2})
    rate = Rate(rate=0.05)
    return option, stock, model, rate


def test_cos_model_type_check(setup):
    option, stock, _, rate = setup
    non_cf_model = MagicMock()
    non_cf_model.supports_cf = False
    non_cf_model.name = "NoCFModel"
    with pytest.raises(TypeError, match="does not support a characteristic function"):
        COSTechnique().price(option, stock, non_cf_model, rate)


@pytest.mark.parametrize("kwargs", [{"width": 0.0}, {"n_terms": 1}])
def test_cos_rejects_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        COSTechnique(**kwargs)


@pytest.mark.parametrize("call", [True, False])
@pytest.mark.parametrize("T", [1 / 365, 0.25, 10.0])
def test_cos_price_and_delta_match_black_scholes(setup, call, T):
    _, stock, model, rate = setup
    option = Option(
        strike=105, maturity=T, option_type=OptionType.CALL if call else OptionType.PUT
    )
    kwargs = dict(spot=100.0, strike=105.0, r=0.05, q=0.01, t=T, call=call)
    technique = COSTechnique()
    assert technique.price(option, stock, model, rate).price == pytest.approx(
        model.price_closed_form(**kwargs), abs=1e-10
    )
    sd = 0.2 * math.sqrt(T)
    d1 = (math.log(100.0 / 105.0) + 0.04 * T) / sd + 0.5 * sd
    delta = math.exp(-0.01 * T) * (0.5 * math.erfc(-d1 / math.sqrt(2)) - (not call))
    assert technique.delta(option, stock, model, rate) == pytest.approx(delta, abs=1e-9)


def test_cos_delta_reuses_the_pricing_series(setup):
    option, stock, model, rate = setup
    technique = COSTechnique()
    technique.price(option, stock, model, rate)
    cached = technique._cached_results["delta"]
    assert technique.delta(option, stock, model, rate) == cached


def test_fixed_number_of_terms_is_used(setup):
    option, stock, model, rate = setup
    coarse = COSTechnique(n_terms=8).price(option, stock, model, rate).price
    fine = COSTechnique().price(option, stock, model, rate).price
    assert abs(coarse - fine) > 1e-6


@pytest.mark.parametrize(
    "model, strip",
    [
        (BSMModel(params={"sigma": 0.2}), (-math.inf, math.inf)),
        (
            MertonJumpModel(params=MertonJumpModel.default_params),
            (-math.inf, math.inf),
        ),
        (
            KouModel(
                params={
                    "sigma": 0.2,
                    "lambda": 1.0,
                    "p_up": 0.3,
                    "eta1": 4.0,
                    "eta2": 9.0,
                }
            ),
            (-9.0, 4.0),
        ),
        (NIGModel(params={"alpha": 15.0, "beta": -5.0, "delta": 0.5}), (-10.0, 20.0)),
        (
            HyperbolicModel(
                params={"alpha": 15.0, "beta": -5.0, "delta": 0.5, "mu": 0.0}
            ),
            (-10.0, 20.0),
        ),
        (CGMYModel(params={"C": 0.5, "G": 5.0, "M": 2.0, "Y": 0.8}), (-5.0, 2.0)),
    ],
)
def test_moment_strip_values(model, strip):
    assert model.moment_strip() == pytest.approx(strip)


def test_variance_gamma_moment_strip_is_where_the_mgf_base_vanishes():
    sigma, nu, theta = 0.2, 0.3, -0.15
    lo, hi = VarianceGammaModel(
        params={"sigma": sigma, "nu": nu, "theta": theta}
    ).moment_strip()
    for s in (lo, hi):
        assert 1 - theta * nu * s - 0.5 * sigma**2 * nu * s**2 == pytest.approx(
            0.0, abs=1e-12
        )
    assert lo < 0 < 1 < hi


def test_moment_strip_keeps_the_truncation_off_kous_pole():
    """
    Regression: with lambda T ~ 7e-4 the analytically continued "moments" past
    Kou's pole at s = -eta2 stay positive and log-convex, so a scan accepted
    them, the Chernoff range collapsed to the diffusion's (-0.35 instead of ~-3)
    and the K=140 call came out at -1.6e-5. References: COS on fixed ranges
    ln S + [-10, 20] and [-20, 40] (N = 2^18, 2^20) with the CF transcribed from
    Kou (2002), and Gil-Pelaez quadrature, agree to 1e-13.
    """
    model = KouModel(
        params={
            "sigma": 0.3125,
            "lambda": 0.03125,
            "p_up": 0.0546875,
            "eta1": 4.0,
            "eta2": 9.0,
        }
    )
    T = 0.021484375
    prices = [
        COSTechnique()
        .price(
            Option(strike=K, maturity=T, option_type=OptionType.CALL),
            Stock(spot=100.0),
            model,
            Rate(rate=0.0),
        )
        .price
        for K in (80.0, 140.0)
    ]
    assert prices == pytest.approx([20.000749423750, 4.5168940781e-4], abs=1e-10)


def test_cos_warns_when_the_series_is_truncated():
    model = CGMYModel(params={"C": 0.0078125, "G": 1.0, "M": 11.0, "Y": 0.0})
    phi = model.cf(t=0.03125, spot=100.0, r=0.0, q=0.0)
    with pytest.warns(RuntimeWarning, match="COS series truncated"):
        cos_prices(
            phi,
            100.0,
            np.array([100.0]),
            np.array([True]),
            0.0,
            0.0,
            0.03125,
            strip=model.moment_strip(),
        )


def test_cos_slow_but_resolved_series_does_not_warn():
    """CGMY with Y = 0 is a difference of gamma processes, i.e. Variance Gamma
    with nu = 1/C, sigma^2 = 2C/(GM), theta = C(1/M - 1/G). |phi| decays only like
    u^(-2CT), so the series runs to max_terms, but the tail bound is small and the
    price matches the normal-mixture truth (put integrated over the gamma
    quantiles, call by parity; Madan, Carr & Chang 1998) to 1e-9. Regression: the
    warning fired whenever |phi(u_N)| > tol."""
    model = CGMYModel(params={"C": 0.3, "G": 6.0, "M": 9.0, "Y": 0.0})
    option = Option(strike=100.0, maturity=1.5, option_type=OptionType.CALL)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        price = COSTechnique().price(option, Stock(spot=100.0), model, Rate(rate=0.05))
    assert price.price == pytest.approx(9.2659572019, abs=1e-9)
