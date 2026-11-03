"""Full-path kernels: shape, initial column, consistency with the terminal
kernels (same generator state -> same terminal values), and SABR forward-to-spot
conversion."""

import math

import numpy as np
import pytest

from optpricing.techniques.kernels import mc_kernels, path_kernels

N_PATHS = 4000
N_STEPS = 20
DT = 0.05
T = N_STEPS * DT
R, Q = 0.05, 0.01
S0 = 100.0
LOG_S0 = math.log(S0)


def gen(seed=0):
    return np.random.default_rng(seed)


@pytest.mark.parametrize(
    "name, args",
    [
        ("bsm", (LOG_S0, R, Q, 0.2, DT)),
        ("heston", (LOG_S0, 0.04, R, Q, 2.0, 0.04, -0.7, 0.3, DT)),
        ("merton", (LOG_S0, R, Q, 0.2, 0.5, -0.1, 0.15, DT)),
        ("bates", (LOG_S0, 0.04, R, Q, 2.0, 0.04, -0.7, 0.3, 0.5, -0.1, 0.15, DT)),
        ("kou", (LOG_S0, R, Q, 0.15, 1.0, 0.6, 10.0, 5.0, DT)),
    ],
)
@pytest.mark.parametrize("antithetic", [False, True])
def test_path_kernel_matches_terminal_kernel(name, args, antithetic):
    """Path and terminal kernels consume the generator identically, so the last
    column of the path matrix must equal the terminal kernel's output."""
    path_kernel = getattr(path_kernels, f"{name}_path_kernel")
    terminal_kernel = getattr(mc_kernels, f"{name}_kernel")
    paths = path_kernel(gen(3), N_PATHS, N_STEPS, antithetic, *args)
    log_terminal = terminal_kernel(gen(3), N_PATHS, N_STEPS, antithetic, *args)

    assert paths.shape == (N_STEPS + 1, N_PATHS)
    np.testing.assert_allclose(paths[0], S0)
    np.testing.assert_allclose(paths[-1], np.exp(log_terminal), rtol=1e-12)
    assert np.all(paths > 0)


def test_bsm_path_kernel_zero_vol_is_pure_drift():
    paths = path_kernels.bsm_path_kernel(
        gen(), 5, N_STEPS, False, LOG_S0, R, Q, 1e-12, DT
    )
    expected = S0 * np.exp((R - Q) * DT * np.arange(N_STEPS + 1))
    np.testing.assert_allclose(paths, np.tile(expected[:, None], (1, 5)), rtol=1e-9)


def test_sabr_path_kernel_returns_spot_paths():
    """The forward is simulated and converted: S_0 equals the spot, and
    E[S_t] tracks the forward at every time step."""
    f0 = S0 * math.exp((R - Q) * T)
    paths = path_kernels.sabr_path_kernel(
        gen(), 40_000, N_STEPS, False, f0, 0.5, R, Q, 0.5, 0.8, -0.6, DT
    )
    np.testing.assert_allclose(paths[0], S0)
    times = DT * np.arange(N_STEPS + 1)
    np.testing.assert_allclose(
        paths.mean(axis=1), S0 * np.exp((R - Q) * times), rtol=6e-3
    )
    assert np.all(paths >= 0.0)


def test_sabr_path_kernel_zero_noise_is_discounted_forward():
    f0 = S0 * math.exp((R - Q) * T)
    paths = path_kernels.sabr_path_kernel(
        gen(), 3, N_STEPS, False, f0, 1e-12, R, Q, 1e-12, 0.8, -0.6, DT
    )
    times = DT * np.arange(N_STEPS + 1)
    np.testing.assert_allclose(
        paths[:, 0], f0 * np.exp(-(R - Q) * (T - times)), rtol=1e-9
    )
    assert paths[0, 0] == pytest.approx(S0)
    assert paths[-1, 0] == pytest.approx(f0)


def test_sabr_jump_path_kernel_shape_and_start():
    f0 = S0 * math.exp((R - Q) * T)
    paths = path_kernels.sabr_jump_path_kernel(
        gen(), 2000, N_STEPS, True, f0, 0.5, R, Q, 0.5, 0.8, -0.6, 0.4, -0.1, 0.15, DT
    )
    assert paths.shape == (N_STEPS + 1, 2000)
    np.testing.assert_allclose(paths[0], S0)
    assert np.all(paths >= 0.0)
