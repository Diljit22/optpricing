"""
Prices against independent references (QuantLib 1.44 and textbook formulas).

See ``conftest.py`` for how the references were generated.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from optpricing import (
    BatesModel,
    BSMModel,
    CEVModel,
    CGMYModel,
    CIRModel,
    HestonModel,
    HyperbolicModel,
    KouModel,
    MertonJumpModel,
    NIGModel,
    Rate,
    Stock,
    VarianceGammaModel,
    VasicekModel,
    ZeroCouponBond,
)
from optpricing.calibration.vectorized_pricer import price_options_vectorized
from optpricing.techniques import (
    AmericanMonteCarloTechnique,
    ClosedFormTechnique,
    COSTechnique,
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


@pytest.mark.parametrize("technique", [IntegrationTechnique, COSTechnique])
@pytest.mark.parametrize("case", HESTON_CASES, ids=[c[0] for c in HESTON_CASES])
def test_heston_cf_techniques_match_quantlib(case, technique):
    _, S, K, T, r, q, v0, kappa, theta, xi, rho, call, ref = case
    model = HestonModel(
        params={"v0": v0, "kappa": kappa, "theta": theta, "rho": rho, "vol_of_vol": xi}
    )
    price = technique().price(
        make_option(K, T, call), Stock(spot=S, dividend=q), model, Rate(rate=r), v0=v0
    )
    assert price.price == pytest.approx(ref, abs=1e-5)


@pytest.mark.parametrize("case", HESTON_CASES, ids=[c[0] for c in HESTON_CASES])
def test_heston_fft_matches_quantlib(case):
    """Carr-Madan FFT with the default settings (measured <= 5e-10; the cubic
    log-strike interpolation used before P5-F17 left a few 1e-4)."""
    _, S, K, T, r, q, v0, kappa, theta, xi, rho, call, ref = case
    model = HestonModel(
        params={"v0": v0, "kappa": kappa, "theta": theta, "rho": rho, "vol_of_vol": xi}
    )
    price = FFTTechnique().price(
        make_option(K, T, call), Stock(spot=S, dividend=q), model, Rate(rate=r), v0=v0
    )
    assert price.price == pytest.approx(ref, abs=1e-9)


# S=100, r=5%, q=2%, calls. Heston (v0 .09, kappa .8, theta .06, xi 1, rho -.9):
# QuantLib 1.44 AnalyticHestonEngine (1e-14). NIG (alpha 6, beta -3, delta .3) and
# hyperbolic (alpha 15, beta -5, delta .5, mu 0): COS method (Fang & Oosterlee
# 2008, L=60, N=2^18) on the papers' characteristic functions, self-checked
# against FO 2008 Tables to 1e-8. Truncating the Gil-Pelaez integrals at u=200
# missed these by 1e-5 (Heston) to 5e-4 (NIG).
SHORT_MATURITY_CASES = [
    ("heston", 10 / 365, 100.0, 1.9945612132),
    ("heston", 10 / 365, 110.0, 0.0028791974),
    ("heston", 37 / 365, 90.0, 11.1975168629),
    ("heston", 37 / 365, 110.0, 0.1849841263),
    ("nig", 0.1, 90.0, 10.8830021736),
    ("nig", 0.1, 100.0, 2.5164395185),
    ("nig", 0.1, 110.0, 0.2861311922),
    ("hyperbolic", 0.1, 100.0, 2.5859547782),
]


def _short_maturity_model(name):
    if name == "heston":
        return HestonModel(
            params={
                "v0": 0.09,
                "kappa": 0.8,
                "theta": 0.06,
                "rho": -0.9,
                "vol_of_vol": 1.0,
            }
        )
    if name == "nig":
        return NIGModel(params={"alpha": 6.0, "beta": -3.0, "delta": 0.3})
    return HyperbolicModel(
        params={"alpha": 15.0, "beta": -5.0, "delta": 0.5, "mu": 0.0}
    )


@pytest.mark.parametrize("case", SHORT_MATURITY_CASES)
def test_quadrature_defaults_resolve_short_maturities(case):
    name, T, K, ref = case
    model = _short_maturity_model(name)
    stock, rate = Stock(spot=100.0, dividend=0.02), Rate(rate=0.05)
    chain_df = pd.DataFrame({"strike": [K], "maturity": [T], "optionType": ["call"]})
    for technique in (IntegrationTechnique(), COSTechnique()):
        scalar = technique.price(make_option(K, T), stock, model, rate).price
        assert scalar == pytest.approx(ref, abs=1e-7)
    for method in ("cos", "gauss"):
        chain = price_options_vectorized(chain_df, stock, model, rate, method=method)
        assert chain[0] == pytest.approx(ref, abs=1e-7)


# S=100, r=5 %, q=2 %, T=1. COS method (Fang & Oosterlee 2008, L=60, N=2^18) on
# characteristic functions transcribed from the original papers: Kou (2002),
# Barndorff-Nielsen (1997) NIG, Eberlein & Keller (1995) hyperbolic. The
# library's quadrature agrees to 1e-13.
LEVY_GOLDEN = [
    ("kou", 80.0, False, 2.5792315012),
    ("kou", 100.0, True, 11.1201033349),
    ("kou", 120.0, True, 3.8718708776),
    ("nig", 80.0, False, 0.9964251935),
    ("nig", 100.0, True, 9.0078271037),
    ("nig", 120.0, True, 2.2884256100),
    ("hyperbolic", 80.0, False, 1.3863761390),
    ("hyperbolic", 100.0, True, 9.8334467955),
    ("hyperbolic", 120.0, True, 2.9339916742),
]


def _levy_golden_model(name):
    if name == "kou":
        return KouModel(
            params={
                "sigma": 0.16,
                "lambda": 1.0,
                "p_up": 0.4,
                "eta1": 10.0,
                "eta2": 5.0,
            }
        )
    if name == "nig":
        return NIGModel(params={"alpha": 15.0, "beta": -5.0, "delta": 0.5})
    return HyperbolicModel(
        params={"alpha": 15.0, "beta": -5.0, "delta": 0.5, "mu": 0.0}
    )


@pytest.mark.parametrize("technique", [IntegrationTechnique, COSTechnique])
@pytest.mark.parametrize("case", LEVY_GOLDEN)
def test_jump_models_match_cos_reference(case, technique):
    name, K, call, ref = case
    price = technique().price(
        make_option(K, 1.0, call),
        Stock(spot=100.0, dividend=0.02),
        _levy_golden_model(name),
        Rate(rate=0.05),
    )
    assert price.price == pytest.approx(ref, abs=1e-8)


# Fang & Oosterlee (2008), SIAM J. Sci. Comput. 31(2), Sections 5.3-5.4:
# S=100, r=0.1, q=0, T=1. VG sigma .12, nu .2, theta -.14, K=90; CGMY C=1,
# G=M=5, K=100 (reference values quoted to 9 decimals).
@pytest.mark.parametrize(
    "model, K, ref",
    [
        (
            VarianceGammaModel(params={"sigma": 0.12, "nu": 0.2, "theta": -0.14}),
            90.0,
            19.099354724,
        ),
        (
            CGMYModel(params={"C": 1.0, "G": 5.0, "M": 5.0, "Y": 0.5}),
            100.0,
            19.812948843,
        ),
        (
            CGMYModel(params={"C": 1.0, "G": 5.0, "M": 5.0, "Y": 1.5}),
            100.0,
            49.790905469,
        ),
    ],
    ids=["VG", "CGMY_Y0.5", "CGMY_Y1.5"],
)
@pytest.mark.parametrize("technique", [IntegrationTechnique, COSTechnique])
def test_levy_models_match_fang_oosterlee_2008(model, K, ref, technique):
    price = technique().price(
        make_option(K, 1.0), Stock(spot=100.0), model, Rate(rate=0.1)
    )
    assert price.price == pytest.approx(ref, abs=2e-9)


# Calls (S=100, r=5 %, q=2 %, T=1; COS references, N=2^16 vs 2^18 agree to 3e-13)
# where E[S_T^(alpha+1)] is finite only for alpha + 1 < 1.5 (Kou eta1), 2 (CGMY M)
# or 2 (NIG alpha - beta): a fixed damping alpha outside that strip returned
# 0.4 for a 37.65 call. Plus Fang & Oosterlee (2008) CGMY Y=1.98 (r=10 %, q=0),
# where alpha=1.75 overflowed to 2.8e+99; its optimal alpha is ~0.02, close to
# the pole of the damped integrand at alpha = 0, which needs the finer eta.
FFT_DAMPING_CASES = [
    ("kou_eta1_1.5", 80.0, 40.8374335674),
    ("kou_eta1_1.5", 100.0, 37.6499945346),
    ("kou_eta1_1.5", 120.0, 35.4826347829),
    ("kou_eta1_2.5", 80.0, 28.3354686996),
    ("kou_eta1_2.5", 100.0, 18.7468184075),
    ("kou_eta1_2.5", 120.0, 13.7061801346),
    ("cgmy_M_2", 80.0, 30.8709118377),
    ("cgmy_M_2", 100.0, 22.5269693894),
    ("cgmy_M_2", 120.0, 17.0607315004),
    ("nig_skewed", 80.0, 28.2334271721),
    ("nig_skewed", 100.0, 19.1995275768),
    ("nig_skewed", 120.0, 13.8026723158),
    ("cgmy_fo_y1.98", 100.0, 99.999905510),
]


def _fft_damping_case(name):
    kou = {"sigma": 0.2, "lambda": 1.0, "p_up": 0.3, "eta2": 4.0}
    if name.startswith("kou"):
        return (
            KouModel(params={**kou, "eta1": float(name.rsplit("_", 1)[1])}),
            0.05,
            0.02,
        )
    if name == "cgmy_M_2":
        return CGMYModel(params={"C": 0.5, "G": 5.0, "M": 2.0, "Y": 0.8}), 0.05, 0.02
    if name == "nig_skewed":
        return NIGModel(params={"alpha": 3.0, "beta": 1.0, "delta": 0.5}), 0.05, 0.02
    return CGMYModel(params={"C": 1.0, "G": 5.0, "M": 5.0, "Y": 1.98}), 0.1, 0.0


@pytest.mark.parametrize("case", FFT_DAMPING_CASES)
def test_fft_damping_stays_inside_the_moment_strip(case):
    name, K, ref = case
    model, r, q = _fft_damping_case(name)
    price = FFTTechnique().price(
        make_option(K, 1.0), Stock(spot=100.0, dividend=q), model, Rate(rate=r)
    )
    assert price.price == pytest.approx(ref, abs=2e-9)


# Short-maturity pure-jump cases (T=0.05, S=100, r=4 %, q=2 %) where the CF
# decays like |u|^(-2T/nu) (VG) or barely at all for small Y (CGMY). Reference:
# COS (Fang & Oosterlee 2008) on fixed ranges ln F -/+ 10, 20, 30 with N = 2^22
# terms, which agree to 5e-8.
SLOW_CF_CASES = [
    ("vg", 80.0, 20.0598904660),
    ("vg", 100.0, 0.6203828500),
    ("vg", 120.0, 0.0388402210),
    ("cgmy", 80.0, 20.0643719820),
    ("cgmy", 100.0, 1.5537320060),
    ("cgmy", 120.0, 0.6653729640),
]


def _slow_cf_model(name):
    if name == "vg":
        return VarianceGammaModel(
            params={"sigma": 0.078057, "nu": 0.644915, "theta": 0.111580}
        )
    return CGMYModel(
        params={"C": 0.652362, "G": 12.114359, "M": 2.708026, "Y": 0.141308}
    )


# Strong Feller violation with rho near -1 (v0 .005, kappa 1, theta .015625,
# rho -.9375, xi 1), S=100, r=q=0, T=91/365: QuantLib 1.44 AnalyticHestonEngine
# and COS agree to 1e-10. Regression (P5-F16): the fixed Gauss-Legendre grid
# gave 29.7713 (below intrinsic) at K=70.
FELLER_CASES = [(70.0, 30.0198214721), (100.0, 0.7643051209), (140.0, 0.0)]


@pytest.mark.parametrize("technique", [IntegrationTechnique, COSTechnique])
@pytest.mark.parametrize("case", FELLER_CASES)
def test_heston_under_strong_feller_violation(case, technique):
    K, ref = case
    params = {
        "v0": 0.005,
        "kappa": 1.0,
        "theta": 0.015625,
        "rho": -0.9375,
        "vol_of_vol": 1.0,
    }
    price = technique().price(
        make_option(K, 91 / 365),
        Stock(spot=100.0),
        HestonModel(params=params),
        Rate(rate=0.0),
        v0=0.005,
    )
    assert price.price == pytest.approx(ref, abs=1e-8)


@pytest.mark.parametrize("case", SLOW_CF_CASES)
def test_cos_with_slowly_decaying_cf(case):
    """|phi| has not decayed at 2^16 terms, which COS reports; the price is
    still within 1e-6 (the density's kink dominates the remaining error)."""
    name, K, ref = case
    with pytest.warns(RuntimeWarning, match="COS series truncated"):
        price = COSTechnique().price(
            make_option(K, 0.05),
            Stock(spot=100.0, dividend=0.02),
            _slow_cf_model(name),
            Rate(rate=0.04),
        )
    assert price.price == pytest.approx(ref, abs=1e-6)


@pytest.mark.parametrize("name", ["vg", "cgmy"])
def test_quadrature_warns_when_cf_decays_too_slowly(name):
    """P5-F16: no Gil-Pelaez grid converges here (errors up to 1.1 on the fixed
    grid, 0.2 with the adaptive guard), so the technique must say so."""
    with pytest.warns(RuntimeWarning, match="COSTechnique"):
        IntegrationTechnique().price(
            make_option(100.0, 0.05),
            Stock(spot=100.0, dividend=0.02),
            _slow_cf_model(name),
            Rate(rate=0.04),
        )


# rho * xi >= kappa (v0 .05, theta .25, rho .5, xi 1, kappa .5 = rho * xi and
# .3 < rho * xi; Bates adds lambda .5, mu_j -.1, sigma_j .15), S=100, r=3 %,
# q=1 %, T=1: QuantLib 1.44 AnalyticHestonEngine / BatesEngine. Regression: the
# CF has a removable 0/0 at u = -i in this regime, so phi(-i) (the P1
# normaliser) was NaN and the quadrature returned NaN prices (found by the
# property tests).
STRONG_LEVERAGE_CASES = [
    ("heston", 0.5, 80.0, False, 1.8355050274),
    ("heston", 0.5, 100.0, True, 10.9939635150),
    ("heston", 0.5, 120.0, True, 6.3203684027),
    ("bates", 0.5, 80.0, False, 2.7215709833),
    ("bates", 0.5, 100.0, True, 12.0593925609),
    ("bates", 0.5, 120.0, True, 6.8112037443),
    ("heston", 0.3, 80.0, False, 1.3993362559),
    ("heston", 0.3, 100.0, True, 9.5471218628),
    ("heston", 0.3, 120.0, True, 5.2909957251),
    ("bates", 0.3, 80.0, False, 2.2656159988),
    ("bates", 0.3, 100.0, True, 10.8071200487),
    ("bates", 0.3, 120.0, True, 5.7509285602),
]


@pytest.mark.parametrize("technique", [IntegrationTechnique, COSTechnique])
@pytest.mark.parametrize("case", STRONG_LEVERAGE_CASES)
def test_stochastic_vol_when_rho_xi_exceeds_kappa(case, technique):
    name, kappa, K, call, ref = case
    params = {"v0": 0.05, "kappa": kappa, "theta": 0.25, "rho": 0.5, "vol_of_vol": 1.0}
    if name == "heston":
        model = HestonModel(params=params)
    else:
        model = BatesModel(
            params={**params, "lambda": 0.5, "mu_j": -0.1, "sigma_j": 0.15}
        )
    price = technique().price(
        make_option(K, 1.0, call),
        Stock(spot=100.0, dividend=0.01),
        model,
        Rate(rate=0.03),
        v0=0.05,
    )
    assert price.price == pytest.approx(ref, abs=1e-7)


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


HULL_DIVIDENDS = ([0.5, 0.5], [2.0 / 12.0, 5.0 / 12.0])


@pytest.mark.parametrize("order", [1, -1], ids=["sorted", "reversed"])
@pytest.mark.parametrize("via", ["stock", "kwargs"])
def test_blacks_approximation_matches_hull(order, via):
    """
    Hull, *Options, Futures and Other Derivatives*, Black's approximation example:
    S = K = 40, r = 9 %, sigma = 30 %, T = 6 months, dividends of 0.5 at 2 and 5
    months. Hold to maturity 3.671233 (Hull: 3.67; BSM on S - PV(dividends),
    scipy), exercise just before the 5-month ex-date 3.524614 (Hull: 3.52),
    before the 2-month one 2.250914. Regressions: unsorted ex-dividend dates
    gave 3.8211,
    and dividends passed as keyword arguments were ignored when the `Stock`
    fields were None.
    """
    from optpricing import BlacksApproxModel

    divs, times = (x[::order] for x in HULL_DIVIDENDS)
    if via == "stock":
        stock = Stock(spot=40.0, discrete_dividends=divs, ex_div_times=times)
        kwargs = {}
    else:
        stock = Stock(spot=40.0)
        kwargs = {"discrete_dividends": divs, "ex_div_times": times}
    price = ClosedFormTechnique().price(
        make_option(40.0, 0.5),
        stock,
        BlacksApproxModel(params={"sigma": 0.3}),
        Rate(rate=0.09),
        **kwargs,
    )
    assert price.price == pytest.approx(3.671233209, abs=1e-8)


@pytest.mark.parametrize("sigma, r", [(0.2, 0.08), (0.4, 0.03)])
def test_perpetual_put_matches_merton_1973(sigma, r):
    """Merton (1973), q = 0: S* = 2rK/(2r + sigma^2) and
    P = (K - S*) (S/S*)^(-2r/sigma^2) above S*."""
    from optpricing import PerpetualPutModel

    K = 100.0
    s_star = 2 * r * K / (2 * r + sigma**2)
    model = PerpetualPutModel(params={"sigma": sigma, "rate": r})
    for S in (s_star * 0.9, s_star * 1.2, 120.0, 200.0):
        ref = (
            K - S if S <= s_star else (K - s_star) * (S / s_star) ** (-2 * r / sigma**2)
        )
        assert model.price_closed_form(spot=S, strike=K, q=0.0) == pytest.approx(
            ref, rel=1e-12
        )


def test_perpetual_put_solves_the_free_boundary_problem():
    """With a dividend yield there is no textbook table; check the defining
    conditions instead: 0.5 sigma^2 S^2 P'' + (r-q) S P' - r P = 0 above the
    boundary, value matching and smooth pasting (P' = -1) at it."""
    from optpricing import PerpetualPutModel

    sigma, r, q, K = 0.25, 0.06, 0.03, 100.0
    model = PerpetualPutModel(params={"sigma": sigma, "rate": r})

    def put(S):
        return model.price_closed_form(spot=S, strike=K, q=q)

    for S in (90.0, 110.0, 150.0):
        h = 1e-3 * S
        d1 = (put(S + h) - put(S - h)) / (2 * h)
        d2 = (put(S + h) - 2 * put(S) + put(S - h)) / h**2
        residual = 0.5 * sigma**2 * S**2 * d2 + (r - q) * S * d1 - r * put(S)
        assert abs(residual) < 1e-5
    b = r - q - 0.5 * sigma**2
    gamma = (-b - math.sqrt(b**2 + 2 * r * sigma**2)) / sigma**2
    s_star = K * gamma / (gamma - 1.0)
    h = 1e-6 * s_star
    assert put(s_star) == pytest.approx(K - s_star, abs=1e-12)
    assert (put(s_star + h) - put(s_star)) / h == pytest.approx(-1.0, abs=1e-5)


def test_merton_series_with_many_jumps_matches_characteristic_function():
    """
    lambda' T ~ 240 needs ~350 series terms. Regression: Poisson weights built
    from factorials overflowed past n = 170 and the closed form returned NaN
    (v2.5.0); the log-space weights agree with the Gil-Pelaez price to 1e-10.
    """
    model = MertonJumpModel(
        params={
            "sigma": 0.1,
            "lambda": 120.0,
            "mu_j": -0.002,
            "sigma_j": 0.01,
            "max_sum_terms": 600,
        }
    )
    args = (make_option(100.0, 2.0), Stock(spot=100.0, dividend=0.01), model)
    closed = ClosedFormTechnique().price(*args, Rate(rate=0.04)).price
    integral = IntegrationTechnique().price(*args, Rate(rate=0.04)).price
    assert closed == pytest.approx(integral, abs=1e-9)


@pytest.mark.parametrize("face", [1.0, 100.0])
@pytest.mark.parametrize("case", ZCB_CASES, ids=[c[0] for c in ZCB_CASES])
def test_zero_coupon_bond_matches_quantlib(case, face):
    """The face value scales the discount bond (it used to be ignored)."""
    _, model_cls, r0, kappa, theta, sigma, T, ref = case
    model = model_cls(params={"kappa": kappa, "theta": theta, "sigma": sigma})
    price = ClosedFormTechnique().price(
        ZeroCouponBond(maturity=T, face_value=face),
        Stock(spot=r0),
        model,
        Rate(rate=0.0),
    )
    assert price.price == pytest.approx(face * ref, abs=face * 1e-9)


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


# Puts near gamma = 1: QuantLib 1.44 AnalyticCEVEngine on the forward with a
# 1-year horizon, mapped to spot dynamics by the time change
# T = ln(1 + 2 mu (gamma-1) tau) / (2 mu (gamma-1)), mu = r - q (S=100, r=5 %,
# q=2 %, sigma = 0.25 * 100^(1-gamma)). The Black-Scholes fallback used for
# |1-gamma| < 1e-3 was off by up to 4e-4 here.
CEV_NEAR_LOGNORMAL = [
    (0.9995, 1.0000150003000068, 80.0, 1.747846901101),
    (0.9995, 1.0000150003000068, 120.0, 20.502209322360),
    (0.9999, 1.0000030000120002, 80.0, 1.747593281594),
    (0.9999, 1.0000030000120002, 100.0, 8.226846487150),
    (0.9999, 1.0000030000120002, 120.0, 20.502510682827),
    (1.0002, 0.9999940000479997, 80.0, 1.747403083985),
    (1.0002, 0.9999940000479997, 120.0, 20.502736713300),
    (1.0005, 0.9999850002999933, 80.0, 1.747212900867),
    (1.0005, 0.9999850002999933, 120.0, 20.502962752339),
]


# Puts for gamma > 1 (strict local martingale; puts are unaffected by the
# martingale-defect convention question for calls): QuantLib 1.44
# AnalyticCEVEngine with the same time change, 1-year horizon, S=100, r=5 %,
# q=2 %.
CEV_ABOVE_ONE = [
    (1.5, 0.025, 0.98529340805148, 80.0, 1.4499031341),
    (1.5, 0.025, 0.98529340805148, 100.0, 8.1863484580),
    (1.5, 0.025, 0.98529340805148, 120.0, 20.8920324825),
    (2.0, 0.0025, 0.9711484687329296, 80.0, 1.1884157620),
    (2.0, 0.0025, 0.9711484687329296, 100.0, 8.1590968475),
    (2.0, 0.0025, 0.9711484687329296, 120.0, 21.3109215026),
]


# Calls for gamma > 1, same set-up plus T = 3 y: QuantLib AnalyticCEVEngine prices
# the expectation E[e^{-rT}(S_T - K)^+]. S is a strict local martingale here, so
# the Hull/Schroder formula exceeds it by the martingale defect (up to 1.98).
CEV_ABOVE_ONE_CALLS = [
    (1.5, 0.025, 0.98529340805148, 80.0, 23.3442735584),
    (1.5, 0.025, 0.98529340805148, 100.0, 11.0421359046),
    (1.5, 0.025, 0.98529340805148, 120.0, 4.7092369515),
    (2.0, 0.0025, 0.9711484687329296, 80.0, 23.0504365927),
    (2.0, 0.0025, 0.9711484687329296, 100.0, 10.9690649578),
    (2.0, 0.0025, 0.9711484687329296, 120.0, 5.0688368926),
    (1.5, 0.025, 2.8725898747017444, 80.0, 29.1706892107),
    (1.5, 0.025, 2.8725898747017444, 100.0, 19.4190584258),
    (1.5, 0.025, 2.8725898747017444, 120.0, 13.0726267168),
    (2.0, 0.0025, 2.7585739746262234, 80.0, 26.4213460386),
    (2.0, 0.0025, 2.7585739746262234, 100.0, 17.1250570242),
    (2.0, 0.0025, 2.7585739746262234, 120.0, 11.5562484252),
]


@pytest.mark.parametrize("case", CEV_ABOVE_ONE_CALLS)
def test_cev_calls_above_gamma_one_are_expectations(case):
    gamma, sigma, T, K, ref = case
    model = CEVModel(params={"sigma": sigma, "gamma": gamma})
    stock, rate = Stock(spot=100.0, dividend=0.02), Rate(rate=0.05)
    price = ClosedFormTechnique().price(make_option(K, T), stock, model, rate)
    assert price.price == pytest.approx(ref, abs=1e-9)
    vectorised = price_options_vectorized(
        pd.DataFrame({"strike": [K], "maturity": [T], "optionType": ["call"]}),
        stock,
        model,
        rate,
    )
    assert vectorised[0] == pytest.approx(ref, abs=1e-9)


@pytest.mark.parametrize("case", CEV_ABOVE_ONE)
def test_cev_puts_above_gamma_one_match_quantlib(case):
    gamma, sigma, T, K, ref = case
    price = ClosedFormTechnique().price(
        make_option(K, T, call=False),
        Stock(spot=100.0, dividend=0.02),
        CEVModel(params={"sigma": sigma, "gamma": gamma}),
        Rate(rate=0.05),
    )
    assert price.price == pytest.approx(ref, abs=1e-9)


@pytest.mark.parametrize("case", CEV_NEAR_LOGNORMAL)
def test_cev_closed_form_near_gamma_one_matches_quantlib(case):
    gamma, T, K, ref = case
    model = CEVModel(params={"sigma": 0.25 * 100.0 ** (1.0 - gamma), "gamma": gamma})
    price = ClosedFormTechnique().price(
        make_option(K, T, call=False),
        Stock(spot=100.0, dividend=0.02),
        model,
        Rate(rate=0.05),
    )
    assert price.price == pytest.approx(ref, abs=1e-7)


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
