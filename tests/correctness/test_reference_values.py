"""
Prices against independent references (QuantLib 1.44 and textbook formulas).

See ``conftest.py`` for how the references were generated.
"""

from __future__ import annotations

import math

import pytest

from optpricing import (
    BatesModel,
    BSMModel,
    CEVModel,
    CIRModel,
    HestonModel,
    MertonJumpModel,
    Rate,
    Stock,
    VasicekModel,
    ZeroCouponBond,
)
from optpricing.techniques import (
    AmericanMonteCarloTechnique,
    ClosedFormTechnique,
    CRRTechnique,
    FFTTechnique,
    IntegrationTechnique,
    LeisenReimerTechnique,
    MonteCarloTechnique,
    PDETechnique,
    TOPMTechnique,
)

from ._helpers import HALF_YEAR, THREE_QUARTERS, make_option

# QuantLib AnalyticHestonEngine (Gauss-Lobatto, 1e-12, 100k evaluations).
HESTON_CASES = [
    # label, S, K, T, r, q, v0, kappa, theta, xi, rho, call, reference
    (
        "atm_call",
        100,
        100,
        1.0,
        0.05,
        0.01,
        0.04,
        2.0,
        0.05,
        0.5,
        -0.7,
        True,
        9.966521164,
    ),
    (
        "atm_put",
        100,
        100,
        1.0,
        0.05,
        0.01,
        0.04,
        2.0,
        0.05,
        0.5,
        -0.7,
        False,
        6.084480239,
    ),
    (
        "feller_ok",
        100,
        100,
        1.0,
        0.05,
        0.01,
        0.04,
        2.0,
        0.05,
        0.3,
        -0.7,
        True,
        10.229453088,
    ),
    (
        "long_T_high_xi",
        100,
        120,
        10.0,
        0.03,
        0.0,
        0.09,
        0.5,
        0.09,
        1.0,
        -0.9,
        True,
        29.018272864,
    ),
    (
        "deep_itm_put",
        100,
        150,
        2.0,
        0.05,
        0.01,
        0.04,
        1.0,
        0.06,
        0.6,
        -0.5,
        False,
        38.965657027,
    ),
]


@pytest.mark.parametrize("case", HESTON_CASES, ids=[c[0] for c in HESTON_CASES])
def test_heston_integration_matches_quantlib(case):
    _, S, K, T, r, q, v0, kappa, theta, xi, rho, call, ref = case
    model = HestonModel(
        params={"v0": v0, "kappa": kappa, "theta": theta, "rho": rho, "vol_of_vol": xi}
    )
    price = IntegrationTechnique().price(
        make_option(K, T, call), Stock(spot=S, dividend=q), model, Rate(rate=r), v0=v0
    )
    assert price.price == pytest.approx(ref, abs=1e-5)


@pytest.mark.parametrize("case", HESTON_CASES, ids=[c[0] for c in HESTON_CASES])
def test_heston_fft_matches_quantlib(case):
    """Carr-Madan FFT with the default grid; linear log-strike interpolation
    limits the accuracy to a few 1e-4."""
    _, S, K, T, r, q, v0, kappa, theta, xi, rho, call, ref = case
    model = HestonModel(
        params={"v0": v0, "kappa": kappa, "theta": theta, "rho": rho, "vol_of_vol": xi}
    )
    price = FFTTechnique().price(
        make_option(K, T, call), Stock(spot=S, dividend=q), model, Rate(rate=r), v0=v0
    )
    assert price.price == pytest.approx(ref, abs=5e-3)


def test_heston_far_otm_short_maturity_is_essentially_zero():
    """QuantLib: 1.0e-7 for K=130, T=0.1; the integrand must not blow up."""
    model = HestonModel(
        params={"v0": 0.04, "kappa": 2.0, "theta": 0.05, "rho": -0.7, "vol_of_vol": 0.5}
    )
    price = IntegrationTechnique().price(
        make_option(130.0, 0.1),
        Stock(spot=100.0, dividend=0.01),
        model,
        Rate(rate=0.05),
        v0=0.04,
    )
    assert abs(price.price) < 1e-4


# Merton (1976) Poisson-weighted BSM series, evaluated independently with 80
# terms; the call also matches QuantLib's BatesEngine with the stochastic
# volatility frozen (xi=1e-3) to 3e-6.
MERTON_CASES = [
    ("atm_call", 100, 100, 1.0, 0.05, 0.01, 0.2, 0.5, -0.1, 0.15, True, 11.027704564),
    (
        "otm_put",
        100,
        80,
        HALF_YEAR,
        0.05,
        0.01,
        0.25,
        1.0,
        -0.2,
        0.3,
        False,
        3.409267048,
    ),
]


@pytest.mark.parametrize("case", MERTON_CASES, ids=[c[0] for c in MERTON_CASES])
@pytest.mark.parametrize("technique", [ClosedFormTechnique, IntegrationTechnique])
def test_merton_matches_series(case, technique):
    _, S, K, T, r, q, sigma, lam, mu_j, sig_j, call, ref = case
    model = MertonJumpModel(
        params={
            "sigma": sigma,
            "lambda": lam,
            "mu_j": mu_j,
            "sigma_j": sig_j,
            "max_sum_terms": 100,
        }
    )
    price = technique().price(
        make_option(K, T, call), Stock(spot=S, dividend=q), model, Rate(rate=r)
    )
    assert price.price == pytest.approx(ref, abs=1e-5)


# QuantLib BatesEngine (1e-12, 100k evaluations).
BATES_CASES = [
    (
        "atm_call",
        100,
        100,
        1.0,
        0.05,
        0.01,
        0.04,
        2.0,
        0.05,
        0.5,
        -0.7,
        0.5,
        -0.1,
        0.15,
        True,
        11.259807425,
    ),
    (
        "otm_put",
        100,
        110,
        THREE_QUARTERS,
        0.05,
        0.01,
        0.04,
        2.0,
        0.05,
        0.3,
        -0.7,
        0.5,
        -0.1,
        0.15,
        False,
        11.748811372,
    ),
]


@pytest.mark.parametrize("case", BATES_CASES, ids=[c[0] for c in BATES_CASES])
def test_bates_integration_matches_quantlib(case):
    _, S, K, T, r, q, v0, kappa, theta, xi, rho, lam, mu_j, sig_j, call, ref = case
    model = BatesModel(
        params={
            "v0": v0,
            "kappa": kappa,
            "theta": theta,
            "rho": rho,
            "vol_of_vol": xi,
            "lambda": lam,
            "mu_j": mu_j,
            "sigma_j": sig_j,
        }
    )
    price = IntegrationTechnique().price(
        make_option(K, T, call), Stock(spot=S, dividend=q), model, Rate(rate=r), v0=v0
    )
    assert price.price == pytest.approx(ref, abs=1e-5)


# QuantLib Vasicek(r0, a, b, sigma).discountBond and
# CoxIngersollRoss(r0, theta, k, sigma).discountBond
ZCB_CASES = [
    ("vasicek_1y", VasicekModel, 0.05, 0.86, 0.09, 0.02, 1.0, 0.9388180337),
    ("vasicek_10y", VasicekModel, 0.03, 0.3, 0.05, 0.015, 10.0, 0.6505138154),
    ("cir_1y", CIRModel, 0.05, 0.86, 0.09, 0.02, 1.0, 0.9387855611),
    ("cir_10y", CIRModel, 0.03, 0.5, 0.04, 0.1, 10.0, 0.6872728726),
]


@pytest.mark.parametrize("case", ZCB_CASES, ids=[c[0] for c in ZCB_CASES])
def test_zero_coupon_bond_matches_quantlib(case):
    _, model_cls, r0, kappa, theta, sigma, T, ref = case
    model = model_cls(params={"kappa": kappa, "theta": theta, "sigma": sigma})
    price = ClosedFormTechnique().price(
        ZeroCouponBond(maturity=T), Stock(spot=r0), model, Rate(rate=0.0)
    )
    assert price.price == pytest.approx(ref, abs=1e-9)


# QuantLib FdBlackScholesVanillaEngine (2000 x 2000); the 2001-step Leisen-Reimer
# engine agrees to 1.6e-4 and 8.5e-5 respectively.
AMERICAN_PUT_CASES = [
    ("atm", 100.0, 100.0, 1.0, 0.05, 0.0, 0.2, 6.0900736382),
    ("itm_with_dividend", 100.0, 110.0, HALF_YEAR, 0.03, 0.02, 0.3, 14.3414602505),
]


@pytest.mark.parametrize(
    "case", AMERICAN_PUT_CASES, ids=[c[0] for c in AMERICAN_PUT_CASES]
)
@pytest.mark.parametrize(
    "technique, tol",
    [
        (lambda: CRRTechnique(steps=1000, is_american=True), 5e-3),
        (lambda: LeisenReimerTechnique(steps=501, is_american=True), 2e-3),
        (lambda: TOPMTechnique(steps=1000, is_american=True), 5e-3),
    ],
    ids=["crr", "leisen_reimer", "topm"],
)
def test_american_put_trees_match_fd_reference(case, technique, tol):
    _, S, K, T, r, q, sigma, ref = case
    price = technique().price(
        make_option(K, T, call=False),
        Stock(spot=S, dividend=q),
        BSMModel(params={"sigma": sigma}),
        Rate(rate=r),
    )
    assert price.price == pytest.approx(ref, abs=tol)


@pytest.mark.parametrize(
    "case", AMERICAN_PUT_CASES, ids=[c[0] for c in AMERICAN_PUT_CASES]
)
def test_american_put_lsmc_matches_fd_reference(case):
    """Longstaff-Schwartz is biased low by O(1e-2) at 50 steps; the regression
    on moneyness-scaled polynomials must keep it within 0.06 of the FD value."""
    _, S, K, T, r, q, sigma, ref = case
    amc = AmericanMonteCarloTechnique(n_paths=100_000, n_steps=50, seed=3, lsm_degree=4)
    price = amc.price(
        make_option(K, T, call=False),
        Stock(spot=S, dividend=q),
        BSMModel(params={"sigma": sigma}),
        Rate(rate=r),
    )
    assert price.price == pytest.approx(ref, abs=0.06)


# Hull (Options, Futures and Other Derivatives, CEV section) closed form for
# dS = (r-q) S dt + sigma S^gamma dW. The r=q=0 case also equals QuantLib's
# AnalyticCEVEngine (forward CEV) to 1e-9.
CEV_CASES = [
    ("gamma_0.5", 100.0, 100.0, 1.0, 0.05, 0.0, 2.0, 0.5, 10.453885329),
    (
        "gamma_0.8_with_dividend",
        100.0,
        95.0,
        HALF_YEAR,
        0.05,
        0.01,
        0.5,
        0.8,
        9.523763683,
    ),
    ("gamma_0.5_zero_rates", 100.0, 100.0, 1.0, 0.0, 0.0, 2.0, 0.5, 7.968853232),
]


@pytest.mark.parametrize("case", CEV_CASES, ids=[c[0] for c in CEV_CASES])
def test_cev_exact_sampler_matches_closed_form(case):
    _, S, K, T, r, q, sigma, gamma, ref = case
    model = CEVModel(params={"sigma": sigma, "gamma": gamma})
    mc = MonteCarloTechnique(n_paths=400_000, seed=11)
    price = mc.price(make_option(K, T), Stock(spot=S, dividend=q), model, Rate(rate=r))
    assert price.price == pytest.approx(ref, abs=0.08)


@pytest.mark.parametrize("case", CEV_CASES, ids=[c[0] for c in CEV_CASES])
def test_cev_exact_sampler_reproduces_forward(case):
    """For gamma < 1 with absorption at zero the spot is a true martingale."""
    import numpy as np

    _, S, _, T, r, q, sigma, gamma, _ = case
    model = CEVModel(params={"sigma": sigma, "gamma": gamma})
    st = model.sample_terminal_spot(S, r, T, 400_000, q=q, rng=np.random.default_rng(5))
    assert st.mean() == pytest.approx(S * math.exp((r - q) * T), rel=3e-3)


def test_pde_price_and_grid_greeks_match_analytic_bsm():
    """Crank-Nicolson with Rannacher start-up; the spot sits on a grid node so
    the grid Greeks are taken at S0 (previously at the node below it)."""
    bsm = BSMModel(params={"sigma": 0.2})
    option = make_option(100.0, 1.0)
    stock, rate = Stock(spot=100.0, dividend=0.01), Rate(rate=0.05)
    cf, pde = ClosedFormTechnique(), PDETechnique(M=200, N=200)

    assert pde.price(option, stock, bsm, rate).price == pytest.approx(
        cf.price(option, stock, bsm, rate).price, abs=1e-2
    )
    assert pde.delta(option, stock, bsm, rate) == pytest.approx(
        cf.delta(option, stock, bsm, rate), abs=5e-4
    )
    assert pde.gamma(option, stock, bsm, rate) == pytest.approx(
        cf.gamma(option, stock, bsm, rate), abs=5e-5
    )


def test_pde_is_second_order_accurate():
    bsm = BSMModel(params={"sigma": 0.2})
    option = make_option(100.0, 1.0, call=False)
    stock, rate = Stock(spot=100.0, dividend=0.01), Rate(rate=0.05)
    exact = ClosedFormTechnique().price(option, stock, bsm, rate).price
    err_coarse = abs(
        PDETechnique(M=100, N=100).price(option, stock, bsm, rate).price - exact
    )
    err_fine = abs(
        PDETechnique(M=400, N=400).price(option, stock, bsm, rate).price - exact
    )
    assert err_fine < err_coarse / 8  # O(h^2) in both S and t: 16x expected


@pytest.mark.parametrize(
    "technique",
    [
        lambda: CRRTechnique(steps=1000),
        lambda: LeisenReimerTechnique(steps=501),
        lambda: TOPMTechnique(steps=1000),
    ],
    ids=["crr", "leisen_reimer", "topm"],
)
def test_european_tree_greeks_match_analytic_bsm(technique):
    bsm = BSMModel(params={"sigma": 0.2})
    option = make_option(100.0, 1.0)
    stock, rate = Stock(spot=100.0, dividend=0.01), Rate(rate=0.05)
    cf, tree = ClosedFormTechnique(), technique()
    assert tree.price(option, stock, bsm, rate).price == pytest.approx(
        cf.price(option, stock, bsm, rate).price, abs=5e-3
    )
    assert tree.delta(option, stock, bsm, rate) == pytest.approx(
        cf.delta(option, stock, bsm, rate), abs=5e-3
    )
    assert tree.gamma(option, stock, bsm, rate) == pytest.approx(
        cf.gamma(option, stock, bsm, rate), abs=5e-4
    )
