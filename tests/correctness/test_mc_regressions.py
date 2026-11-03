"""
Monte Carlo estimators against characteristic-function prices and external
references. Each test pins a bug that was found in the Phase 2 audit.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from optpricing import (
    BatesModel,
    BSMModel,
    CEVModel,
    CGMYModel,
    HestonModel,
    HyperbolicModel,
    NIGModel,
    VarianceGammaModel,
)
from optpricing.models.base import sample_from_cf
from optpricing.techniques import (
    AmericanMonteCarloTechnique,
    IntegrationTechnique,
    MonteCarloTechnique,
)

from ._helpers import make_option


def test_heston_mc_correlation_is_not_applied_twice(stock, rate, atm_call, heston_base):
    """
    Regression: the European MC driver pre-correlated the Brownian increments
    while the kernels correlate them again (effective rho ~ -0.92 for rho=-0.7),
    giving 9.78 against QuantLib's 9.9665 (AnalyticHestonEngine).
    """
    mc = MonteCarloTechnique(n_paths=200_000, n_steps=252, seed=1)
    price = mc.price(atm_call, stock, heston_base, rate, v0=0.04).price
    assert price == pytest.approx(9.966521164, abs=0.08)


def test_heston_mc_is_insensitive_to_rho_sign_at_the_money_forward(stock, rate):
    """Sanity: the leverage effect moves OTM puts up and OTM calls down."""
    otm_put = make_option(85.0, 1.0, call=False)
    neg = HestonModel(
        params={"v0": 0.04, "kappa": 2.0, "theta": 0.04, "rho": -0.7, "vol_of_vol": 0.3}
    )
    pos = HestonModel(
        params={"v0": 0.04, "kappa": 2.0, "theta": 0.04, "rho": 0.7, "vol_of_vol": 0.3}
    )
    p_neg = (
        MonteCarloTechnique(n_paths=200_000, n_steps=100, seed=2)
        .price(otm_put, stock, neg, rate, v0=0.04)
        .price
    )
    p_pos = (
        MonteCarloTechnique(n_paths=200_000, n_steps=100, seed=2)
        .price(otm_put, stock, pos, rate, v0=0.04)
        .price
    )
    assert p_neg > p_pos


def test_bates_mc_matches_quantlib(stock, rate, atm_call):
    """QuantLib BatesEngine: 11.259807."""
    bates = BatesModel(
        params={
            "v0": 0.04,
            "kappa": 2.0,
            "theta": 0.05,
            "rho": -0.7,
            "vol_of_vol": 0.5,
            "lambda": 0.5,
            "mu_j": -0.1,
            "sigma_j": 0.15,
        }
    )
    mc = MonteCarloTechnique(n_paths=200_000, n_steps=252, seed=1)
    assert mc.price(atm_call, stock, bates, rate, v0=0.04).price == pytest.approx(
        11.259807425, abs=0.1
    )


def test_nig_exact_sampler_matches_characteristic_function(stock, rate, atm_call):
    """
    Regression: the inverse-Gaussian subordinator was drawn with scipy's default
    shape (1) instead of (delta T)^2, biasing NIG MC by -0.29 on a 9.63 option.
    """
    nig = NIGModel(params={"alpha": 15.0, "beta": -5.0, "delta": 0.5})
    ref = IntegrationTechnique().price(atm_call, stock, nig, rate).price
    mc = (
        MonteCarloTechnique(n_paths=400_000, seed=1)
        .price(atm_call, stock, nig, rate)
        .price
    )
    assert mc == pytest.approx(ref, abs=0.05)


def test_nig_sampler_moments():
    alpha, beta, delta, T = 15.0, -5.0, 0.5, 0.25
    gamma = math.sqrt(alpha**2 - beta**2)
    x = NIGModel(
        params={"alpha": alpha, "beta": beta, "delta": delta}
    ).sample_terminal_log_return(T, 1_000_000, np.random.default_rng(0))
    assert x.mean() == pytest.approx(delta * T * beta / gamma, abs=5e-4)
    assert x.var() == pytest.approx(delta * T * alpha**2 / gamma**3, rel=1e-2)


def test_variance_gamma_exact_sampler_matches_characteristic_function(
    stock, rate, atm_call
):
    vg = VarianceGammaModel(params={"sigma": 0.2, "nu": 0.1, "theta": -0.14})
    ref = IntegrationTechnique().price(atm_call, stock, vg, rate).price
    mc = (
        MonteCarloTechnique(n_paths=400_000, seed=1)
        .price(atm_call, stock, vg, rate)
        .price
    )
    assert mc == pytest.approx(ref, abs=0.05)


@pytest.mark.parametrize(
    "y", [0.0, 0.5, 1.0, 1.2], ids=["Y=0 (gamma difference)", "Y=0.5", "Y=1", "Y=1.2"]
)
def test_cgmy_mc_matches_characteristic_function(stock, rate, y):
    """
    Regression: sampling was guarded on Y=1 but implemented the Y=0 (gamma
    difference) law, and the CF had Gamma(-Y) poles at Y=0 and Y=1.
    """
    cgmy = CGMYModel(params={"C": 0.3, "G": 6.0, "M": 9.0, "Y": y})
    option = make_option(100.0, 1.5)
    ref = IntegrationTechnique().price(option, stock, cgmy, rate).price
    mc = (
        MonteCarloTechnique(n_paths=400_000, seed=1)
        .price(option, stock, cgmy, rate)
        .price
    )
    assert mc == pytest.approx(ref, abs=0.06)


@pytest.mark.parametrize("T", [0.5, 1.0, 2.0])
def test_hyperbolic_mc_matches_characteristic_function(stock, rate, T):
    """
    Regression: the CF mixed the horizon-T hyperbolic law with a t-th power,
    the drift was not a martingale correction for T != 1, and the sampler used
    scipy's dimensionless shape parameters as if they were alpha and beta.
    """
    hyp = HyperbolicModel(params={"alpha": 15.0, "beta": -5.0, "delta": 0.5, "mu": 0.0})
    option = make_option(100.0, T)
    ref = IntegrationTechnique().price(option, stock, hyp, rate).price
    mc = (
        MonteCarloTechnique(n_paths=400_000, seed=1)
        .price(option, stock, hyp, rate)
        .price
    )
    assert mc == pytest.approx(ref, abs=0.06)


def test_cf_inverse_transform_sampler_recovers_a_normal_law():
    mu, sd = 0.3, 0.7
    x = sample_from_cf(
        lambda u: np.exp(1j * u * mu - 0.5 * sd**2 * u**2),
        400_000,
        np.random.default_rng(0),
    )
    assert x.mean() == pytest.approx(mu, abs=5e-3)
    assert x.std() == pytest.approx(sd, rel=1e-2)
    assert np.mean(x < mu) == pytest.approx(0.5, abs=3e-3)


def test_cf_inverse_transform_sampler_agrees_with_exact_vg_sampler():
    vg = VarianceGammaModel(params={"sigma": 0.2, "nu": 0.3, "theta": -0.14})
    exact = vg.sample_terminal_log_return(0.75, 400_000, np.random.default_rng(1))
    inverted = sample_from_cf(vg.raw_cf(t=0.75), 400_000, np.random.default_rng(2))
    assert inverted.mean() == pytest.approx(exact.mean(), abs=2e-3)
    assert inverted.std() == pytest.approx(exact.std(), rel=1.5e-2)
    assert np.quantile(inverted, 0.05) == pytest.approx(
        np.quantile(exact, 0.05), abs=3e-3
    )


def test_cev_sampler_is_reproducible_and_respects_dividends():
    """
    Regression: the previous sampler used the global numpy RNG, ignored the
    dividend yield and used a wrong noncentral chi-square parametrisation
    (E[S_T] = 100.6 against a forward of 105.1).
    """
    cev = CEVModel(params={"sigma": 2.0, "gamma": 0.5})
    a = cev.sample_terminal_spot(
        100.0, 0.05, 1.0, 50_000, q=0.03, rng=np.random.default_rng(7)
    )
    b = cev.sample_terminal_spot(
        100.0, 0.05, 1.0, 50_000, q=0.03, rng=np.random.default_rng(7)
    )
    np.testing.assert_array_equal(a, b)
    assert a.mean() == pytest.approx(100.0 * math.exp(0.02), rel=1e-2)


def test_cev_zero_rates_do_not_divide_by_zero():
    cev = CEVModel(params={"sigma": 2.0, "gamma": 0.5})
    st = cev.sample_terminal_spot(
        100.0, 0.0, 1.0, 200_000, rng=np.random.default_rng(0)
    )
    assert np.all(np.isfinite(st))
    assert st.mean() == pytest.approx(100.0, rel=1e-2)


def test_cev_absorption_probability_matches_theory():
    """gamma=0.5 is a squared Bessel process of dimension 0; extinction
    probability by time T is exp(-X_0 / (2 T)) with X_0 = S_0 / sigma^2 * 4."""
    sigma, S0, T = 5.0, 100.0, 5.0
    cev = CEVModel(params={"sigma": sigma, "gamma": 0.5})
    st = cev.sample_terminal_spot(S0, 0.0, T, 400_000, rng=np.random.default_rng(3))
    x0 = S0 / (0.25 * sigma**2)
    assert np.mean(st == 0.0) == pytest.approx(math.exp(-x0 / (2 * T)), abs=3e-3)


def test_lsmc_regression_is_well_conditioned_at_high_degree(stock, rate):
    """
    Regression: the Longstaff-Schwartz regression solved the normal equations
    with an explicit inverse on raw powers of S (up to ~1e10 for degree 5) and
    swallowed failures with a bare except. Reference: QuantLib FD 6.0900736
    (q=0 case) — here with q=1% the put is worth slightly more, so compare
    across degrees instead of against the q=0 reference.
    """
    put = make_option(100.0, 1.0, call=False)
    bsm = BSMModel(params={"sigma": 0.2})
    prices = [
        AmericanMonteCarloTechnique(n_paths=60_000, n_steps=50, seed=4, lsm_degree=d)
        .price(put, stock, bsm, rate)
        .price
        for d in (2, 3, 5)
    ]
    assert max(prices) - min(prices) < 0.1
    assert all(np.isfinite(prices))


def test_sabr_kernel_terminal_spots_are_non_negative():
    """
    Regression: only the diffusion coefficient was floored, so a large
    down-move left the spot negative for the rest of the path (2.2% of paths,
    minimum -207 on this parameter set) and puts paid K - S > K. Zero is
    absorbing for 0 < beta < 1.
    """
    from optpricing.techniques.kernels import sabr_kernel

    n, m, dt = 50_000, 50, 2.0 / 50
    rng = np.random.default_rng(9)
    dw1 = rng.standard_normal((n, m)) * math.sqrt(dt)
    dw2 = rng.standard_normal((n, m)) * math.sqrt(dt)
    terminal = sabr_kernel(
        n_paths=n,
        n_steps=m,
        s0=100.0,
        v0=3.0,
        r=0.05,
        q=0.01,
        nu=1.5,
        beta=0.3,
        rho=-0.5,
        dt=dt,
        dw1=dw1,
        dw2=dw2,
    )
    assert np.all(terminal >= 0.0)
    assert np.mean(terminal == 0.0) > 0.0  # the absorbing state is actually reached


@pytest.mark.parametrize(
    "params, strike, maturity, call, tol",
    [
        ({"alpha": 0.25, "beta": 1.0, "rho": -0.3, "nu": 0.3}, 100.0, 1.0, True, 0.03),
        ({"alpha": 2.5, "beta": 0.5, "rho": 0.0, "nu": 0.4}, 90.0, 0.5, False, 0.03),
        ({"alpha": 0.5, "beta": 0.8, "rho": -0.6, "nu": 0.5}, 120.0, 1.0, True, 0.05),
    ],
    ids=["lognormal", "beta_half_put", "skewed_otm_call"],
)
def test_sabr_forward_mc_matches_hagan_black(
    stock, rate, params, strike, maturity, call, tol
):
    """
    The kernels simulate Hagan's forward dynamics, so Monte Carlo must agree
    with Black-76 + Hagan's expansion up to the expansion's own O(nu^2 T) error
    (largest for the skewed, high vol-of-vol case).
    """
    from optpricing import SABRModel
    from optpricing.techniques import ClosedFormTechnique

    sabr = SABRModel(params=params)
    option = make_option(strike, maturity, call)
    hagan = ClosedFormTechnique().price(option, stock, sabr, rate).price
    mc = (
        MonteCarloTechnique(n_paths=400_000, n_steps=200, seed=1)
        .price(option, stock, sabr, rate)
        .price
    )
    assert mc == pytest.approx(hagan, abs=tol)


def test_sabr_path_kernel_returns_spot_paths():
    """The American driver needs spot paths; the kernel evolves the forward and
    discounts it back: S_0 must equal the spot and E[S_t] the forward at t."""
    from optpricing.techniques.kernels import sabr_path_kernel

    n, steps, T, r, q = 20_000, 50, 1.0, 0.05, 0.01
    dt = T / steps
    rng = np.random.default_rng(0)
    dw1 = rng.standard_normal((n, steps)) * math.sqrt(dt)
    dw2 = rng.standard_normal((n, steps)) * math.sqrt(dt)
    f0 = 100.0 * math.exp((r - q) * T)
    paths = sabr_path_kernel(n, steps, f0, 0.5, r, q, 0.5, 0.8, -0.6, dt, dw1, dw2)
    assert paths[:, 0] == pytest.approx(100.0)
    assert paths[:, 25].mean() == pytest.approx(
        100.0 * math.exp((r - q) * 0.5), rel=5e-3
    )


def test_sabr_american_put_dominates_european(stock, rate):
    from optpricing import SABRModel
    from optpricing.techniques import ClosedFormTechnique

    sabr = SABRModel(params={"alpha": 0.5, "beta": 0.8, "rho": -0.6, "nu": 0.5})
    put = make_option(100.0, 1.0, call=False)
    european = ClosedFormTechnique().price(put, stock, sabr, rate).price
    american = (
        AmericanMonteCarloTechnique(n_paths=60_000, n_steps=50, seed=3, lsm_degree=3)
        .price(put, stock, sabr, rate)
        .price
    )
    assert american >= european - 0.02


def test_heston_and_bates_defaults_satisfy_feller(recwarn):
    import warnings

    from optpricing import BatesModel, HestonModel

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a Feller warning would raise here
        heston, bates = HestonModel(), BatesModel()
    for p in (heston.params, bates.params):
        assert 2 * p["kappa"] * p["theta"] >= p["vol_of_vol"] ** 2
