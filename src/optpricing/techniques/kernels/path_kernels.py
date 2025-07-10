from __future__ import annotations

import numba
import numpy as np

__doc__ = """
JIT-compiled kernel for American option pricing via Monte Carlo.
"""


# Docstring for bsm_path_kernel
"""
JIT-compiled kernel for BSM SDE that returns the full path matrix.
"""


@numba.jit(nopython=True, fastmath=True, cache=True)
def bsm_path_kernel(
    n_paths: int,
    n_steps: int,
    log_s0: float,
    r: float,
    q: float,
    sigma: float,
    dt: float,
    dw: np.ndarray,
) -> np.ndarray:
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = np.exp(log_s0)
    log_s = np.full(n_paths, log_s0)
    drift = (r - q - 0.5 * sigma**2) * dt
    for i in range(n_steps):
        log_s += drift + sigma * dw[:, i]
        paths[:, i + 1] = np.exp(log_s)
    return paths


# Docstring for heston_path_kernel
"""
JIT-compiled kernel for Heston SDE that returns the full path matrix.
"""


@numba.jit(nopython=True, fastmath=True, cache=True)
def heston_path_kernel(
    n_paths: int,
    n_steps: int,
    log_s0: float,
    v0: float,
    r: float,
    q: float,
    kappa: float,
    theta: float,
    rho: float,
    vol_of_vol: float,
    dt: float,
    dw1: np.ndarray,
    dw2: np.ndarray,
) -> np.ndarray:
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = np.exp(log_s0)
    log_s = np.full(n_paths, log_s0)
    v = np.full(n_paths, v0)
    rho_bar = np.sqrt(1 - rho**2)

    for i in range(n_steps):
        z1, z2 = dw1[:, i], dw2[:, i]
        correlated_z2 = rho * z1 + rho_bar * z2
        v_pos = np.maximum(v, 0)
        v_sqrt = np.sqrt(v_pos)

        log_s += (r - q - 0.5 * v_pos) * dt + v_sqrt * z1
        v = np.maximum(
            0, v + kappa * (theta - v_pos) * dt + vol_of_vol * v_sqrt * correlated_z2
        )
        paths[:, i + 1] = np.exp(log_s)
    return paths


# Docstring for merton_path_kernel
"""
JIT-compiled kernel for Merton SDE that returns the full path matrix.
"""


@numba.jit(nopython=True, fastmath=True, cache=True)
def merton_path_kernel(
    n_paths: int,
    n_steps: int,
    log_s0: float,
    r: float,
    q: float,
    sigma: float,
    lambda_: float,
    mu_j: float,
    sigma_j: float,
    dt: float,
    dw: np.ndarray,
    jump_counts: np.ndarray,
) -> np.ndarray:
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = np.exp(log_s0)
    log_s = np.full(n_paths, log_s0)
    compensator = lambda_ * (np.exp(mu_j + 0.5 * sigma_j**2) - 1)
    drift = (r - q - 0.5 * sigma**2 - compensator) * dt

    for i in range(n_steps):
        log_s += drift + sigma * dw[:, i]
        jumps_this_step = jump_counts[:, i]
        if np.any(jumps_this_step > 0):
            num_jumps = np.sum(jumps_this_step)
            jump_sizes = np.random.normal(mu_j, sigma_j, num_jumps)
            jump_idx = 0
            for path_idx in range(n_paths):
                if jumps_this_step[path_idx] > 0:
                    log_s[path_idx] += np.sum(
                        jump_sizes[jump_idx : jump_idx + jumps_this_step[path_idx]]
                    )
                    jump_idx += jumps_this_step[path_idx]
        paths[:, i + 1] = np.exp(log_s)
    return paths


# Docstring for bates_path_kernel
"""
JIT-compiled kernel for Bates SDE that returns the full path matrix.
"""


@numba.jit(nopython=True, fastmath=True, cache=True)
def bates_path_kernel(
    n_paths: int,
    n_steps: int,
    log_s0: float,
    v0: float,
    r: float,
    q: float,
    kappa: float,
    theta: float,
    rho: float,
    vol_of_vol: float,
    lambda_: float,
    mu_j: float,
    sigma_j: float,
    dt: float,
    dw1: np.ndarray,
    dw2: np.ndarray,
    jump_counts: np.ndarray,
) -> np.ndarray:
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = np.exp(log_s0)
    log_s = np.full(n_paths, log_s0)
    v = np.full(n_paths, v0)
    compensator = lambda_ * (np.exp(mu_j + 0.5 * sigma_j**2) - 1)
    rho_bar = np.sqrt(1 - rho**2)

    for i in range(n_steps):
        z1, z2 = dw1[:, i], dw2[:, i]
        correlated_z2 = rho * z1 + rho_bar * z2
        v_pos = np.maximum(v, 0)
        v_sqrt = np.sqrt(v_pos)

        log_s += (r - q - 0.5 * v_pos - compensator) * dt + v_sqrt * z1
        v = np.maximum(
            0, v + kappa * (theta - v_pos) * dt + vol_of_vol * v_sqrt * correlated_z2
        )

        jumps_this_step = jump_counts[:, i]
        if np.any(jumps_this_step > 0):
            num_jumps = np.sum(jumps_this_step)
            jump_sizes = np.random.normal(mu_j, sigma_j, num_jumps)
            jump_idx = 0
            for path_idx in range(n_paths):
                if jumps_this_step[path_idx] > 0:
                    log_s[path_idx] += np.sum(
                        jump_sizes[jump_idx : jump_idx + jumps_this_step[path_idx]]
                    )
                    jump_idx += jumps_this_step[path_idx]
        paths[:, i + 1] = np.exp(log_s)
    return paths


# Docstring for sabr_path_kernel
"""
JIT-compiled kernel for SABR SDE that returns the full path matrix.
"""


@numba.jit(nopython=True, fastmath=True, cache=True)
def sabr_path_kernel(
    n_paths: int,
    n_steps: int,
    s0: float,
    v0: float,
    r: float,
    q: float,
    alpha: float,
    beta: float,
    rho: float,
    dt: float,
    dw1: np.ndarray,
    dw2: np.ndarray,
) -> np.ndarray:
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = s0
    s = np.full(n_paths, s0)
    v = np.full(n_paths, v0)
    rho_bar = np.sqrt(1 - rho**2)

    for i in range(n_steps):
        z1, z2 = dw1[:, i], dw2[:, i]
        correlated_z2 = rho * z1 + rho_bar * z2
        s_pos, v_pos = np.maximum(s, 1e-8), np.maximum(v, 0)

        s += (r - q) * s_pos * dt + v_pos * (s_pos**beta) * z1
        v = v * np.exp(-0.5 * alpha**2 * dt + alpha * correlated_z2)
        paths[:, i + 1] = s
    return paths


# Docstring for sabr_jump_path_kernel
"""
JIT-compiled kernel for SABR Jump SDE that returns the full path matrix.
"""


@numba.jit(nopython=True, fastmath=True, cache=True)
def sabr_jump_path_kernel(
    n_paths: int,
    n_steps: int,
    s0: float,
    v0: float,
    r: float,
    q: float,
    alpha: float,
    beta: float,
    rho: float,
    lambda_: float,
    mu_j: float,
    sigma_j: float,
    dt: float,
    dw1: np.ndarray,
    dw2: np.ndarray,
    jump_counts: np.ndarray,
) -> np.ndarray:
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = s0
    s = np.full(n_paths, s0)
    v = np.full(n_paths, v0)
    rho_bar = np.sqrt(1 - rho**2)
    compensator = lambda_ * (np.exp(mu_j + 0.5 * sigma_j**2) - 1)

    for i in range(n_steps):
        z1, z2 = dw1[:, i], dw2[:, i]
        correlated_z2 = rho * z1 + rho_bar * z2
        s_pos, v_pos = np.maximum(s, 1e-8), np.maximum(v, 0)

        s += (r - q - compensator) * s_pos * dt + v_pos * (s_pos**beta) * z1
        v = v * np.exp(-0.5 * alpha**2 * dt + alpha * correlated_z2)

        jumps_this_step = jump_counts[:, i]
        if np.any(jumps_this_step > 0):
            num_jumps = np.sum(jumps_this_step)
            jump_multipliers = np.exp(np.random.normal(mu_j, sigma_j, num_jumps))
            jump_idx = 0
            for path_idx in range(n_paths):
                if jumps_this_step[path_idx] > 0:
                    for _ in range(jumps_this_step[path_idx]):
                        s[path_idx] *= jump_multipliers[jump_idx]
                        jump_idx += 1
        paths[:, i + 1] = s
    return paths


# Docstring for kou_path_kernel
"""
JIT-compiled kernel for Kou SDE that returns the full path matrix.
"""


@numba.jit(nopython=True, fastmath=True, cache=True)
def kou_path_kernel(
    n_paths: int,
    n_steps: int,
    log_s0: float,
    r: float,
    q: float,
    sigma: float,
    lambda_: float,
    p_up: float,
    eta1: float,
    eta2: float,
    dt: float,
    dw: np.ndarray,
    jump_counts: np.ndarray,
) -> np.ndarray:
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = np.exp(log_s0)
    log_s = np.full(n_paths, log_s0)
    compensator = lambda_ * (
        (p_up * eta1 / (eta1 - 1)) + ((1 - p_up) * eta2 / (eta2 + 1)) - 1
    )
    drift = (r - q - 0.5 * sigma**2 - compensator) * dt

    for i in range(n_steps):
        log_s += drift + sigma * dw[:, i]
        jumps_this_step = jump_counts[:, i]
        if np.any(jumps_this_step > 0):
            num_total_jumps = np.sum(jumps_this_step)
            up_or_down = np.random.random(num_total_jumps)
            up_jumps = np.random.exponential(1.0 / eta1, num_total_jumps)
            down_jumps = -np.random.exponential(1.0 / eta2, num_total_jumps)
            all_jumps = np.where(up_or_down < p_up, up_jumps, down_jumps)

            jump_idx = 0
            for path_idx in range(n_paths):
                num_jumps_on_path = jumps_this_step[path_idx]
                if num_jumps_on_path > 0:
                    log_s[path_idx] += np.sum(
                        all_jumps[jump_idx : jump_idx + num_jumps_on_path]
                    )
                    jump_idx += num_jumps_on_path
        paths[:, i + 1] = np.exp(log_s)
    return paths
