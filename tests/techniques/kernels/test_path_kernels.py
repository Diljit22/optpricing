import numpy as np
import pytest

from optpricing.techniques.kernels import path_kernels

N_PATHS = 10
N_STEPS = 1
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
