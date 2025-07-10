import numpy as np
import pytest
from scipy import stats

from optpricing.techniques.kernels import path_kernels

N_PATHS = 10
N_STEPS = 1
N_PATHS_STOCHASTIC = 50000
DT = 0.01
R = 0.05
Q = 0.01


def test_bsm_path_kernel_drift():
    """Tests the deterministic drift of the BSM path kernel."""
    s0 = 100.0
    log_s0 = np.log(s0)
    sigma = 0.2
    dw = np.zeros((N_PATHS, N_STEPS))

    paths = path_kernels.bsm_path_kernel(
        N_PATHS,
        N_STEPS,
        log_s0,
        R,
        Q,
        sigma,
        DT,
        dw,
    )

    expected_drift = (R - Q - 0.5 * sigma**2) * DT
    assert paths[0, -1] == pytest.approx(s0 * np.exp(expected_drift))


def test_heston_path_kernel_drift():
    """Tests the deterministic drift of the Heston path kernel."""
    s0, v0 = 100.0, 0.04
    log_s0 = np.log(s0)
    kappa, theta, rho, vol_of_vol = 2.0, 0.04, -0.7, 0.5
    dw1 = np.zeros((N_PATHS, N_STEPS))
    dw2 = np.zeros((N_PATHS, N_STEPS))

    paths = path_kernels.heston_path_kernel(
        N_PATHS,
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
    assert paths[0, -1] == pytest.approx(s0 * np.exp(expected_s_drift))


def test_merton_path_kernel_drift():
    """Tests the deterministic drift of the Merton path kernel (no jumps)."""
    s0 = 100.0
    log_s0 = np.log(s0)
    sigma, lambda_, mu_j, sigma_j = 0.2, 0.5, -0.1, 0.15
    dw = np.zeros((N_PATHS, N_STEPS))
    jump_counts = np.zeros((N_PATHS, N_STEPS), dtype=np.int64)

    paths = path_kernels.merton_path_kernel(
        N_PATHS,
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
    assert paths[0, -1] == pytest.approx(s0 * np.exp(expected_drift))


def test_bates_path_kernel_drift():
    """Tests the deterministic drift of the Bates path kernel (no jumps)."""
    s0, v0 = 100.0, 0.04
    log_s0 = np.log(s0)
    kappa, theta, rho, vol_of_vol = 2.0, 0.04, -0.7, 0.5
    lambda_, mu_j, sigma_j = 0.5, -0.1, 0.15
    dw1 = np.zeros((N_PATHS, N_STEPS))
    dw2 = np.zeros((N_PATHS, N_STEPS))
    jump_counts = np.zeros((N_PATHS, N_STEPS), dtype=np.int64)

    paths = path_kernels.bates_path_kernel(
        N_PATHS,
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
    assert paths[0, -1] == pytest.approx(s0 * np.exp(expected_s_drift))


def test_sabr_path_kernel_drift():
    """Tests the deterministic drift of the SABR path kernel."""
    s0, v0 = 100.0, 0.5
    alpha, beta, rho = 0.5, 0.8, -0.6
    dw1 = np.zeros((N_PATHS, N_STEPS))
    dw2 = np.zeros((N_PATHS, N_STEPS))

    paths = path_kernels.sabr_path_kernel(
        N_PATHS,
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
    assert paths[0, -1] == pytest.approx(s0 + expected_s_drift)


def test_bsm_path_kernel_statistical_properties():
    """
    Tests that the terminal values of the BSM path kernel match the
    known analytical distribution of log-prices.
    """
    s0 = 100.0
    log_s0 = np.log(s0)
    sigma = 0.2
    T = N_STEPS * DT

    rng = np.random.default_rng(0)
    dw = rng.normal(0, np.sqrt(DT), (N_PATHS_STOCHASTIC, N_STEPS))

    paths = path_kernels.bsm_path_kernel(
        N_PATHS_STOCHASTIC,
        N_STEPS,
        log_s0,
        R,
        Q,
        sigma,
        DT,
        dw,
    )

    sT = paths[:, -1]
    log_sT = np.log(sT)

    expected_mean = log_s0 + (R - Q - 0.5 * sigma**2) * T
    expected_variance = sigma**2 * T

    # Mean test
    t_stat, p_value_mean = stats.ttest_1samp(log_sT, expected_mean)
    msg_ = f"Mean test failed: p-value {p_value_mean:.4f} is too low."
    assert p_value_mean > 0.01, msg_

    # Variance test
    actual_variance = np.var(log_sT)
    chi2_stat = (N_PATHS_STOCHASTIC - 1) * actual_variance / expected_variance
    p_value_var = stats.chi2.sf(chi2_stat, N_PATHS_STOCHASTIC - 1) * 2  # Two-tailed
    msg_ = f"Variance test failed: p-value {p_value_var:.4f} is too low."
    assert p_value_var > 0.01, msg_


def test_heston_path_kernel_risk_neutral_mean():
    """
    Tests that the terminal values of the Heston path kernel follow the
    risk-neutral expectation E[S_T] = S_0 * exp((r-q)*T).
    """
    s0, v0 = 100.0, 0.04
    log_s0 = np.log(s0)
    kappa, theta, rho, vol_of_vol = 2.0, 0.04, -0.7, 0.5
    T = N_STEPS * DT

    rng = np.random.default_rng(0)
    dw1 = rng.normal(0, np.sqrt(DT), (N_PATHS_STOCHASTIC, N_STEPS))
    dw2 = rng.normal(0, np.sqrt(DT), (N_PATHS_STOCHASTIC, N_STEPS))

    paths = path_kernels.heston_path_kernel(
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
        DT,
        dw1,
        dw2,
    )

    sT = paths[:, -1]
    expected_mean_spot = s0 * np.exp((R - Q) * T)

    t_stat, p_value = stats.ttest_1samp(sT, expected_mean_spot)
    msg_ = f"Heston path mean spot price test failed: p-value {p_value:.4f}."
    assert p_value > 0.01, msg_


def test_merton_path_kernel_jumps_are_applied():
    """
    Tests that the Merton path kernel correctly applies jumps by comparing
    a simulation with jumps to one without.
    """
    s0 = 100.0
    log_s0 = np.log(s0)
    sigma, lambda_, mu_j, sigma_j = 0.2, 5.0, -0.1, 0.15

    rng = np.random.default_rng(0)
    dw = rng.normal(0, np.sqrt(DT), (N_PATHS_STOCHASTIC, N_STEPS))

    # With jumps
    jump_counts = rng.poisson(lambda_ * DT, (N_PATHS_STOCHASTIC, N_STEPS))
    paths_with_jumps = path_kernels.merton_path_kernel(
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

    # Without jumps
    no_jump_counts = np.zeros((N_PATHS_STOCHASTIC, N_STEPS), dtype=np.int64)
    paths_no_jumps = path_kernels.merton_path_kernel(
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

    # The means of the terminal prices should be different
    sT_with_jumps = paths_with_jumps[:, -1]
    sT_no_jumps = paths_no_jumps[:, -1]

    t_stat, p_value = stats.ttest_ind(sT_with_jumps, sT_no_jumps, equal_var=False)
    msg_ = "Merton path distributions should be different, but were not."
    assert p_value < 0.01, msg_
