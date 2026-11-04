"""Array-valued closed forms (strike and call/put vectors) agree with the
scalar formulas, and the new CEV closed form matches its references."""

from __future__ import annotations

import math

import numpy as np
import pytest

from optpricing.atoms import Option, OptionType, Rate, Stock
from optpricing.models import BSMModel, CEVModel, MertonJumpModel, SABRModel
from optpricing.techniques import ClosedFormTechnique, MonteCarloTechnique

STRIKES = np.array([70.0, 85.0, 100.0, 115.0, 140.0])
IS_CALL = np.array([True, False, True, False, True])
PRICING = {"spot": 100.0, "r": 0.03, "q": 0.015, "t": 0.75}


@pytest.mark.parametrize(
    "model",
    [
        BSMModel(params={"sigma": 0.25}),
        MertonJumpModel(params=MertonJumpModel.default_params),
        SABRModel(),
        CEVModel(),
    ],
    ids=lambda m: m.name,
)
def test_vectorized_closed_form_matches_scalar(model):
    assert model.vectorized_closed_form
    vec = model.price_closed_form(strike=STRIKES, call=IS_CALL, **PRICING)
    assert isinstance(vec, np.ndarray) and vec.shape == STRIKES.shape
    scalar = [
        model.price_closed_form(strike=float(k), call=bool(c), **PRICING)
        for k, c in zip(STRIKES, IS_CALL)
    ]
    np.testing.assert_allclose(vec, scalar, rtol=1e-12, atol=1e-12)
    # scalar inputs still return a Python/NumPy scalar
    assert np.ndim(model.price_closed_form(strike=100.0, **PRICING)) == 0


# Hull (Options, Futures and Other Derivatives, CEV section) closed form for
# dS = (r-q) S dt + sigma S^gamma dW; the r=q=0 case also equals QuantLib 1.44's
# AnalyticCEVEngine (forward CEV) to 1e-9. Same cases as the Phase 2 sampler tests.
CEV_CASES = [
    ("gamma_0.5", 100.0, 100.0, 1.0, 0.05, 0.0, 2.0, 0.5, 10.453885329),
    (
        "gamma_0.8_with_dividend",
        100.0,
        95.0,
        183 / 365,
        0.05,
        0.01,
        0.5,
        0.8,
        9.523763683,
    ),
    ("gamma_0.5_zero_rates", 100.0, 100.0, 1.0, 0.0, 0.0, 2.0, 0.5, 7.968853232),
]


@pytest.mark.parametrize("case", CEV_CASES, ids=[c[0] for c in CEV_CASES])
def test_cev_closed_form_matches_hull_reference(case):
    _, S, K, T, r, q, sigma, gamma, ref = case
    model = CEVModel(params={"sigma": sigma, "gamma": gamma})
    call = model.price_closed_form(spot=S, strike=K, r=r, q=q, t=T, call=True)
    put = model.price_closed_form(spot=S, strike=K, r=r, q=q, t=T, call=False)
    assert call == pytest.approx(ref, abs=1e-8)
    # the absorbed CEV spot is a martingale, so parity holds exactly
    assert call - put == pytest.approx(S * math.exp(-q * T) - K * math.exp(-r * T))


def test_cev_closed_form_gamma_above_one_matches_exact_sampler():
    model = CEVModel(params={"sigma": 0.2 / 100**0.5, "gamma": 1.5})
    stock, rate = Stock(spot=100.0, dividend=0.01), Rate(rate=0.05)
    option = Option(strike=100.0, maturity=1.0, option_type=OptionType.CALL)
    cf = ClosedFormTechnique().price(option, stock, model, rate).price
    mc = MonteCarloTechnique(n_paths=400_000, seed=11).price(option, stock, model, rate)
    assert cf == pytest.approx(mc.price, abs=0.08)


def test_cev_closed_form_reduces_to_bsm_at_gamma_one():
    cev = CEVModel(params={"sigma": 0.2, "gamma": 1.0})
    bsm = BSMModel(params={"sigma": 0.2})
    kwargs = {"spot": 100.0, "strike": 105.0, "r": 0.05, "q": 0.0, "t": 1.0}
    assert cev.price_closed_form(**kwargs) == pytest.approx(
        bsm.price_closed_form(**kwargs), rel=1e-12
    )
    # continuity across the switch to the lognormal branch
    near = CEVModel(params={"sigma": 0.2 * 100**0.002, "gamma": 0.998})
    assert near.price_closed_form(**kwargs) == pytest.approx(
        bsm.price_closed_form(**kwargs), rel=2e-3
    )


def test_cev_is_now_priced_in_closed_form_by_the_fastest_technique():
    from optpricing.calibration import select_fastest_technique

    assert CEVModel.has_closed_form
    assert isinstance(select_fastest_technique(CEVModel()), ClosedFormTechnique)
