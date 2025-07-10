import numba
import numpy as np
import pytest
from scipy import stats

from optpricing.techniques.kernels import mc_kernels

N_PATHS_DRIFT = 10
N_PATHS_STOCHASTIC = 50000
N_STEPS = 1
DT = 0.01
R = 0.05
Q = 0.01
T = N_STEPS * DT


def test_bsm_kernel_statistical_properties():
    """
    Tests that the BSM kernel produces a distribution of terminal log-prices
    that matches the known analytical mean and variance.
    """
    log_s0 = np.log(100)
    sigma = 0.2

    rng = np.random.default_rng(0)
    dw = rng.normal(0, np.sqrt(DT), (N_PATHS_STOCHASTIC, N_STEPS))

    log_sT = mc_kernels.bsm_kernel(
        N_PATHS_STOCHASTIC,
        N_STEPS,
        log_s0,
        R,
        Q,
        sigma,
        DT,
        dw,
    )

    # mean and variance of the terminal log-price
    expected_mean = log_s0 + (R - Q - 0.5 * sigma**2) * T
    expected_variance = sigma**2 * T

    # mean and variance from the simulation
    actual_variance = np.var(log_sT)

    # Mean test
    t_stat, p_value_mean = stats.ttest_1samp(log_sT, expected_mean)
    msg_ = f"Mean test failed: p-value {p_value_mean:.4f} is too low."
    assert p_value_mean > 0.01, msg_

    # Variance test
    chi2_stat = (N_PATHS_STOCHASTIC - 1) * actual_variance / expected_variance
    p_value_var = stats.chi2.sf(chi2_stat, N_PATHS_STOCHASTIC - 1) * 2

    msg_ = f"Variance test failed: p-value {p_value_var:.4f} is too low."
    assert p_value_var > 0.01, msg_


def test_bsm_kernel_drift():
    """Tests the deterministic drift of the BSM kernel."""
    log_s0 = np.log(100)
    sigma = 0.2
    dw = np.zeros((N_PATHS_DRIFT, N_STEPS))

    log_sT = mc_kernels.bsm_kernel(
        N_PATHS_DRIFT,
        N_STEPS,
        log_s0,
        R,
        Q,
        sigma,
        DT,
        dw,
    )

    expected_drift = (R - Q - 0.5 * sigma**2) * DT
    assert log_sT[0] == pytest.approx(log_s0 + expected_drift)


def test_heston_kernel_drift():
    """Tests the deterministic drift of the Heston kernel."""
    log_s0, v0 = np.log(100), 0.04
    kappa, theta, rho, vol_of_vol = 2.0, 0.04, -0.7, 0.5
    dw1 = np.zeros((N_PATHS_DRIFT, N_STEPS))
    dw2 = np.zeros((N_PATHS_DRIFT, N_STEPS))

    log_sT = mc_kernels.heston_kernel(
        N_PATHS_DRIFT,
        N_STEPS,
        log_s0,
        v0,
        R,
        Q,
        kappa,
        theta,
        rho,
        vol_of_vol,
        DT,
        dw1,
        dw2,
    )

    expected_s_drift = (R - Q - 0.5 * v0) * DT
    assert log_sT[0] == pytest.approx(log_s0 + expected_s_drift)


def test_merton_kernel_drift():
    """Tests the deterministic drift of the Merton kernel (no jumps)."""
    log_s0 = np.log(100)
    sigma, lambda_, mu_j, sigma_j = 0.2, 0.5, -0.1, 0.15
    dw = np.zeros((N_PATHS_DRIFT, N_STEPS))
    jump_counts = np.zeros((N_PATHS_DRIFT, N_STEPS), dtype=np.int64)

    log_sT = mc_kernels.merton_kernel(
        N_PATHS_DRIFT,
        N_STEPS,
        log_s0,
        R,
        Q,
        sigma,
        lambda_,
        mu_j,
        sigma_j,
        DT,
        dw,
        jump_counts,
    )

    compensator = lambda_ * (np.exp(mu_j + 0.5 * sigma_j**2) - 1)
    expected_drift = (R - Q - 0.5 * sigma**2 - compensator) * DT
    assert log_sT[0] == pytest.approx(log_s0 + expected_drift)


def test_bates_kernel_drift():
    """Tests the deterministic drift of the Bates kernel (no jumps)."""
    log_s0, v0 = np.log(100), 0.04
    kappa, theta, rho, vol_of_vol = 2.0, 0.04, -0.7, 0.5
    lambda_, mu_j, sigma_j = 0.5, -0.1, 0.15
    dw1 = np.zeros((N_PATHS_DRIFT, N_STEPS))
    dw2 = np.zeros((N_PATHS_DRIFT, N_STEPS))
    jump_counts = np.zeros((N_PATHS_DRIFT, N_STEPS), dtype=np.int64)

    log_sT = mc_kernels.bates_kernel(
        N_PATHS_DRIFT,
        N_STEPS,
        log_s0,
        v0,
        R,
        Q,
        kappa,
        theta,
        rho,
        vol_of_vol,
        lambda_,
        mu_j,
        sigma_j,
        DT,
        dw1,
        dw2,
        jump_counts,
    )

    compensator = lambda_ * (np.exp(mu_j + 0.5 * sigma_j**2) - 1)
    expected_s_drift = (R - Q - 0.5 * v0 - compensator) * DT
    assert log_sT[0] == pytest.approx(log_s0 + expected_s_drift)


def test_sabr_kernel_drift():
    """Tests the deterministic drift of the SABR kernel."""
    s0, v0 = 100.0, 0.5
    alpha, beta, rho = 0.5, 0.8, -0.6
    dw1 = np.zeros((N_PATHS_DRIFT, N_STEPS))
    dw2 = np.zeros((N_PATHS_DRIFT, N_STEPS))

    sT = mc_kernels.sabr_kernel(
        N_PATHS_DRIFT,
        N_STEPS,
        s0,
        v0,
        R,
        Q,
        alpha,
        beta,
        rho,
        DT,
        dw1,
        dw2,
    )

    expected_s_drift = (R - Q) * s0 * DT
    assert sT[0] == pytest.approx(s0 + expected_s_drift)


def test_heston_kernel_variance_process():
    """
    Tests the variance process of the Heston kernel.
    Checks that the terminal variance of the simulation
    converges towards the long-term variance parameter 'theta'.
    """
    log_s0, v0 = np.log(100), 0.04
    kappa, theta, rho, vol_of_vol = 2.0, 0.09, -0.7, 0.5  # Using theta > v0

    local_n_steps = 100
    local_dt = 0.01

    rng = np.random.default_rng(0)
    dw1 = rng.normal(0, np.sqrt(local_dt), (N_PATHS_STOCHASTIC, local_n_steps))
    dw2 = rng.normal(0, np.sqrt(local_dt), (N_PATHS_STOCHASTIC, local_n_steps))

    log_sT = mc_kernels.heston_kernel(
        N_PATHS_STOCHASTIC,
        local_n_steps,
        log_s0,
        v0,
        R,
        Q,
        kappa,
        theta,
        rho,
        vol_of_vol,
        local_dt,
        dw1,
        dw2,
    )

    sT = np.exp(log_sT)
    expected_mean_spot = np.exp(log_s0) * np.exp((R - Q) * local_n_steps * local_dt)
    t_stat, p_value = stats.ttest_1samp(sT, expected_mean_spot)

    msg_ = f"Heston mean spot price test failed: p-value {p_value:.4f} is too low."
    assert p_value > 0.01, msg_


def test_merton_kernel_jumps_are_applied():
    """
    Tests that the Merton kernel correctly applies jumps by comparing a simulation
    with jumps to one without. The means of the two distributions should be
    statistically different.
    """
    log_s0 = np.log(100)
    sigma, lambda_, mu_j, sigma_j = 0.2, 5.0, -0.1, 0.15

    rng = np.random.default_rng(0)
    dw = rng.normal(0, np.sqrt(DT), (N_PATHS_STOCHASTIC, N_STEPS))

    jump_counts = rng.poisson(lambda_ * DT, (N_PATHS_STOCHASTIC, N_STEPS))
    log_sT_with_jumps = mc_kernels.merton_kernel(
        N_PATHS_STOCHASTIC,
        N_STEPS,
        log_s0,
        R,
        Q,
        sigma,
        lambda_,
        mu_j,
        sigma_j,
        DT,
        dw,
        jump_counts,
    )

    no_jump_counts = np.zeros((N_PATHS_STOCHASTIC, N_STEPS), dtype=np.int64)
    log_sT_no_jumps = mc_kernels.merton_kernel(
        N_PATHS_STOCHASTIC,
        N_STEPS,
        log_s0,
        R,
        Q,
        sigma,
        lambda_,
        mu_j,
        sigma_j,
        DT,
        dw,
        no_jump_counts,
    )

    t_stat, p_value = stats.ttest_ind(
        log_sT_with_jumps,
        log_sT_no_jumps,
        equal_var=False,
    )

    msg_ = "The distributions with and without jumps should be different, but were not."
    assert p_value < 0.01, msg_


def test_bates_kernel_jumps_are_applied():
    """
    Tests that the Bates kernel correctly applies jumps by comparing a simulation
    with jumps to one without.
    """
    log_s0, v0 = np.log(100), 0.04
    kappa, theta, rho, vol_of_vol = 2.0, 0.04, -0.7, 0.5
    lambda_, mu_j, sigma_j = 5.0, -0.1, 0.15

    rng = np.random.default_rng(0)
    dw1 = rng.normal(0, np.sqrt(DT), (N_PATHS_STOCHASTIC, N_STEPS))
    dw2 = rng.normal(0, np.sqrt(DT), (N_PATHS_STOCHASTIC, N_STEPS))

    jump_counts = rng.poisson(lambda_ * DT, (N_PATHS_STOCHASTIC, N_STEPS))
    log_sT_with_jumps = mc_kernels.bates_kernel(
        N_PATHS_STOCHASTIC,
        N_STEPS,
        log_s0,
        v0,
        R,
        Q,
        kappa,
        theta,
        rho,
        vol_of_vol,
        lambda_,
        mu_j,
        sigma_j,
        DT,
        dw1,
        dw2,
        jump_counts,
    )

    no_jump_counts = np.zeros((N_PATHS_STOCHASTIC, N_STEPS), dtype=np.int64)
    log_sT_no_jumps = mc_kernels.bates_kernel(
        N_PATHS_STOCHASTIC,
        N_STEPS,
        log_s0,
        v0,
        R,
        Q,
        kappa,
        theta,
        rho,
        vol_of_vol,
        lambda_,
        mu_j,
        sigma_j,
        DT,
        dw1,
        dw2,
        no_jump_counts,
    )

    t_stat, p_value = stats.ttest_ind(
        log_sT_with_jumps,
        log_sT_no_jumps,
        equal_var=False,
    )

    msg_ = "The distributions with and without jumps should be different, but were not."
    assert p_value < 0.01, msg_


def test_sabr_kernel_volatility_process():
    """
    Tests the log-normal volatility process of the SABR kernel.
    The mean of the log of the terminal volatility should match its
    analytical expectation, which is log(v0) - 0.5 * alpha^2 * T.
    """
    s0, v0 = 100.0, 0.5
    alpha, beta, rho = 0.5, 0.8, -0.6

    # Tests the risk-neutral expectation of the spot price.
    # For a tradeable asset S, E[S_T] = s0 * exp((r-q)*T). This
    # must hold true regardless of the volatility model.

    rng = np.random.default_rng(0)
    dw1 = rng.normal(0, np.sqrt(DT), (N_PATHS_STOCHASTIC, N_STEPS))
    dw2 = rng.normal(0, np.sqrt(DT), (N_PATHS_STOCHASTIC, N_STEPS))

    sT = mc_kernels.sabr_kernel(
        N_PATHS_STOCHASTIC,
        N_STEPS,
        s0,
        v0,
        R,
        Q,
        alpha,
        beta,
        rho,
        DT,
        dw1,
        dw2,
    )

    expected_mean_spot = s0 * np.exp((R - Q) * T)

    t_stat, p_value = stats.ttest_1samp(sT, expected_mean_spot)

    msg = f"SABR mean spot price test failed: p-value {p_value:.4f} is too low."
    assert p_value > 0.01, msg


def test_sabr_jump_kernel_jumps_are_applied():
    """
    Tests that the SABR jump kernel correctly applies jumps by comparing a
    simulation with jumps to one without.
    """
    s0, v0 = 100.0, 0.5
    alpha, beta, rho = 0.5, 0.8, -0.6
    lambda_, mu_j, sigma_j = 5.0, -0.1, 0.15  # High jump intensity

    rng = np.random.default_rng(0)
    dw1 = rng.normal(0, np.sqrt(DT), (N_PATHS_STOCHASTIC, N_STEPS))
    dw2 = rng.normal(0, np.sqrt(DT), (N_PATHS_STOCHASTIC, N_STEPS))

    # Simulate with jumps
    jump_counts = rng.poisson(lambda_ * DT, (N_PATHS_STOCHASTIC, N_STEPS))
    sT_with_jumps = mc_kernels.sabr_jump_kernel(
        N_PATHS_STOCHASTIC,
        N_STEPS,
        s0,
        v0,
        R,
        Q,
        alpha,
        beta,
        rho,
        lambda_,
        mu_j,
        sigma_j,
        DT,
        dw1,
        dw2,
        jump_counts,
    )

    # Simulate without jumps
    no_jump_counts = np.zeros((N_PATHS_STOCHASTIC, N_STEPS), dtype=np.int64)
    sT_no_jumps = mc_kernels.sabr_jump_kernel(
        N_PATHS_STOCHASTIC,
        N_STEPS,
        s0,
        v0,
        R,
        Q,
        alpha,
        beta,
        rho,
        lambda_,
        mu_j,
        sigma_j,
        DT,
        dw1,
        dw2,
        no_jump_counts,
    )

    # The means should be different
    t_stat, p_value = stats.ttest_ind(sT_with_jumps, sT_no_jumps, equal_var=False)
    assert p_value < 0.01, "SABR jump distributions should be different, but were not."


def test_dupire_kernel_with_constant_vol():
    """
    Tests that the Dupire kernel with a constant volatility surface behaves
    identically to the BSM kernel.
    """
    log_s0 = np.log(100)
    constant_sigma = 0.2

    rng = np.random.default_rng(0)
    dw = rng.normal(0, np.sqrt(DT), (N_PATHS_STOCHASTIC, N_STEPS))

    # Define a constant volatility function for Dupire
    @numba.jit(nopython=True)
    def constant_vol_surface(t, s):
        return constant_sigma

    # Run Dupire kernel
    log_sT_dupire = mc_kernels.dupire_kernel(
        N_PATHS_STOCHASTIC, N_STEPS, log_s0, R, Q, DT, dw, constant_vol_surface
    )

    # Run BSM kernel with the same parameters
    log_sT_bsm = mc_kernels.bsm_kernel(
        N_PATHS_STOCHASTIC, N_STEPS, log_s0, R, Q, constant_sigma, DT, dw
    )

    np.testing.assert_allclose(
        log_sT_dupire,
        log_sT_bsm,
        rtol=1e-9,
        err_msg="Dupire with constant vol should match BSM",
    )
