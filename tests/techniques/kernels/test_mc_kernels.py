"""Terminal-value kernels: forward (martingale) property, antithetic structure,
distributional checks and reproducibility."""

import math

import numba
import numpy as np
import pytest

from optpricing.techniques.kernels import mc_kernels

N_PATHS = 100_000
N_STEPS = 20
DT = 0.02
T = N_STEPS * DT
R, Q = 0.05, 0.01
LOG_S0 = math.log(100.0)
FORWARD = 100.0 * math.exp((R - Q) * T)


def gen(seed=0):
    return np.random.default_rng(seed)


def _forward_check(terminal_spot, label, rel=3e-3):
    assert terminal_spot.mean() == pytest.approx(FORWARD, rel=rel), label


def test_bsm_kernel_log_price_moments():
    sigma = 0.2
    log_st = mc_kernels.bsm_kernel(
        gen(), N_PATHS, N_STEPS, False, LOG_S0, R, Q, sigma, DT
    )
    assert log_st.mean() == pytest.approx(
        LOG_S0 + (R - Q - 0.5 * sigma**2) * T, abs=3e-3
    )
    assert log_st.var() == pytest.approx(sigma**2 * T, rel=2e-2)


def test_bsm_kernel_zero_vol_is_pure_drift():
    log_st = mc_kernels.bsm_kernel(gen(), 10, N_STEPS, False, LOG_S0, R, Q, 1e-12, DT)
    np.testing.assert_allclose(log_st, LOG_S0 + (R - Q) * T, rtol=1e-12)


def test_antithetic_halves_are_mirror_images():
    """With antithetic=True the second half uses the negated increments: for
    BSM the log-returns of path j and path j + n/2 sum to twice the drift."""
    sigma = 0.3
    log_st = mc_kernels.bsm_kernel(gen(), 1000, N_STEPS, True, LOG_S0, R, Q, sigma, DT)
    first, second = log_st[:500] - LOG_S0, log_st[500:] - LOG_S0
    np.testing.assert_allclose(
        first + second, 2 * (R - Q - 0.5 * sigma**2) * T, atol=1e-12
    )


@pytest.mark.parametrize("antithetic", [False, True])
def test_heston_kernel_is_a_martingale(antithetic):
    log_st = mc_kernels.heston_kernel(
        gen(),
        N_PATHS,
        N_STEPS,
        antithetic,
        LOG_S0,
        0.04,
        R,
        Q,
        2.0,
        0.04,
        -0.7,
        0.3,
        DT,
    )
    _forward_check(np.exp(log_st), "heston")


def test_merton_kernel_is_a_martingale_and_jumps_add_variance():
    sigma, lam, mu_j, sig_j = 0.2, 1.0, -0.1, 0.2
    log_st = mc_kernels.merton_kernel(
        gen(), N_PATHS, N_STEPS, False, LOG_S0, R, Q, sigma, lam, mu_j, sig_j, DT
    )
    _forward_check(np.exp(log_st), "merton", rel=4e-3)
    expected_var = sigma**2 * T + lam * T * (mu_j**2 + sig_j**2)
    assert log_st.var() == pytest.approx(expected_var, rel=3e-2)


def test_bates_kernel_is_a_martingale():
    log_st = mc_kernels.bates_kernel(
        gen(),
        N_PATHS,
        N_STEPS,
        False,
        LOG_S0,
        0.04,
        R,
        Q,
        2.0,
        0.04,
        -0.7,
        0.3,
        0.5,
        -0.1,
        0.15,
        DT,
    )
    _forward_check(np.exp(log_st), "bates", rel=4e-3)


def test_kou_kernel_is_a_martingale_and_jumps_add_variance():
    sigma, lam, p_up, eta1, eta2 = 0.15, 1.0, 0.6, 10.0, 5.0
    log_st = mc_kernels.kou_kernel(
        gen(), N_PATHS, N_STEPS, False, LOG_S0, R, Q, sigma, lam, p_up, eta1, eta2, DT
    )
    _forward_check(np.exp(log_st), "kou", rel=4e-3)
    jump_second_moment = p_up * 2 / eta1**2 + (1 - p_up) * 2 / eta2**2
    assert log_st.var() == pytest.approx(
        sigma**2 * T + lam * T * jump_second_moment, rel=3e-2
    )


def test_sabr_kernel_forward_is_a_martingale_and_non_negative():
    f0, v0, nu, beta, rho = FORWARD, 0.5, 0.5, 0.8, -0.6
    ft = mc_kernels.sabr_kernel(
        gen(), N_PATHS, N_STEPS, False, f0, v0, nu, beta, rho, DT
    )
    assert ft.mean() == pytest.approx(f0, rel=3e-3)
    assert np.all(ft >= 0.0)


def test_sabr_kernel_zero_vol_of_vol_zero_noise_is_constant():
    ft = mc_kernels.sabr_kernel(
        gen(), 10, N_STEPS, False, 100.0, 1e-12, 1e-12, 0.8, -0.6, DT
    )
    np.testing.assert_allclose(ft, 100.0, rtol=1e-9)


def test_sabr_jump_kernel_forward_is_a_martingale():
    ft = mc_kernels.sabr_jump_kernel(
        gen(),
        N_PATHS,
        N_STEPS,
        False,
        FORWARD,
        0.5,
        0.5,
        0.8,
        -0.6,
        0.4,
        -0.1,
        0.15,
        DT,
    )
    assert ft.mean() == pytest.approx(FORWARD, rel=5e-3)


def test_dupire_kernel_with_constant_vol_matches_bsm():
    sigma = 0.2

    @numba.njit
    def constant_vol_surface(t, s):
        return sigma

    log_dupire = mc_kernels.dupire_kernel(
        gen(1), 5000, N_STEPS, True, LOG_S0, R, Q, DT, constant_vol_surface
    )
    log_bsm = mc_kernels.bsm_kernel(
        gen(1), 5000, N_STEPS, True, LOG_S0, R, Q, sigma, DT
    )
    np.testing.assert_allclose(log_dupire, log_bsm, rtol=1e-12)


def test_kernels_are_reproducible_for_a_seed():
    a = mc_kernels.heston_kernel(
        gen(42), 2000, 5, True, LOG_S0, 0.04, R, Q, 2.0, 0.04, -0.7, 0.3, DT
    )
    b = mc_kernels.heston_kernel(
        gen(42), 2000, 5, True, LOG_S0, 0.04, R, Q, 2.0, 0.04, -0.7, 0.3, DT
    )
    np.testing.assert_array_equal(a, b)
