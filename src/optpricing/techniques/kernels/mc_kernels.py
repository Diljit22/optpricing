from __future__ import annotations

import numba
import numpy as np

__doc__ = """
This module contains JIT-compiled (`numba`) kernels for simulating the SDE
paths of different financial models. These functions are designed for performance.
"""

# Docstring for bsm_kernel
"""
JIT-compiled kernel for BSM SDE simulation.
d(logS) = (r - q - 0.5*sigma^2)dt + sigma*dW
"""


@numba.jit(nopython=True, fastmath=True, cache=True)
def bsm_kernel(
    n_paths,
    n_steps,
    log_s0,
    r,
    q,
    sigma,
    dt,
    dw,
):
    log_s = np.full(n_paths, log_s0)
    drift = (r - q - 0.5 * sigma**2) * dt
    for i in range(n_steps):
        log_s += drift + sigma * dw[:, i]

    return log_s


# Docstring for heston_kernel
"""
JIT-compiled kernel for Heston SDE simulation (Full Truncation).
"""


@numba.jit(nopython=True, fastmath=True, cache=True)
def heston_kernel(
    n_paths,
    n_steps,
    log_s0,
    v0,
    r,
    q,
    kappa,
    theta,
    rho,
    vol_of_vol,
    dt,
    dw1,
    dw2,
):
    log_s = np.full(n_paths, log_s0)
    v = np.full(n_paths, v0)
    rho_bar = np.sqrt(1 - rho**2)

    for i in range(n_steps):
        z1 = dw1[:, i]
        z2 = dw2[:, i]

        correlated_z2 = rho * z1 + rho_bar * z2

        v_pos = np.maximum(v, 0)
        v_sqrt = np.sqrt(v_pos)

        # Evolve log-spot with the first independent draw
        log_s += (r - q - 0.5 * v_pos) * dt + v_sqrt * z1

        # Evolve variance with the correlated draw
        v += kappa * (theta - v_pos) * dt + vol_of_vol * v_sqrt * correlated_z2
        v = np.maximum(v, 0)  # Apply reflection to variance

    return log_s


# Docstring for merton_kernel
"""
JIT-compiled kernel for Merton Jump-Diffusion SDE simulation.
"""


@numba.jit(nopython=True, fastmath=True, cache=True)
def merton_kernel(
    n_paths,
    n_steps,
    log_s0,
    r,
    q,
    sigma,
    lambda_,
    mu_j,
    sigma_j,
    dt,
    dw,
    jump_counts,
):
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
                for _ in range(jumps_this_step[path_idx]):
                    log_s[path_idx] += jump_sizes[jump_idx]
                    jump_idx += 1

    return log_s


# Docstring for bates_kernel
"""
JIT-compiled kernel for Bates (Heston + Jumps) SDE simulation.
"""


@numba.jit(nopython=True, fastmath=True, cache=True)
def bates_kernel(
    n_paths,
    n_steps,
    log_s0,
    v0,
    r,
    q,
    kappa,
    theta,
    rho,
    vol_of_vol,
    lambda_,
    mu_j,
    sigma_j,
    dt,
    dw1,
    dw2,
    jump_counts,
):
    log_s = np.full(n_paths, log_s0)
    v = np.full(n_paths, v0)
    compensator = lambda_ * (np.exp(mu_j + 0.5 * sigma_j**2) - 1)
    rho_bar = np.sqrt(1 - rho**2)

    for i in range(n_steps):
        z1 = dw1[:, i]
        z2 = dw2[:, i]

        correlated_z2 = rho * z1 + rho_bar * z2

        v_pos = np.maximum(v, 0)
        v_sqrt = np.sqrt(v_pos)

        # Heston part
        log_s += (r - q - 0.5 * v_pos - compensator) * dt + v_sqrt * z1
        v += kappa * (theta - v_pos) * dt + vol_of_vol * v_sqrt * correlated_z2
        v = np.maximum(v, 0)

        # Merton Jump part
        jumps_this_step = jump_counts[:, i]
        if np.any(jumps_this_step > 0):
            num_jumps = np.sum(jumps_this_step)
            jump_sizes = np.random.normal(mu_j, sigma_j, num_jumps)
            jump_idx = 0
            for path_idx in range(n_paths):
                for _ in range(jumps_this_step[path_idx]):
                    log_s[path_idx] += jump_sizes[jump_idx]
                    jump_idx += 1

    return log_s


# Docstring for sabr_kernel
"""
JIT-compiled kernel for SABR SDE simulation.
"""


@numba.jit(nopython=True, fastmath=True, cache=True)
def sabr_kernel(
    n_paths,
    n_steps,
    s0,
    v0,
    r,
    q,
    alpha,
    beta,
    rho,
    dt,
    dw1,
    dw2,
):
    s = np.full(n_paths, s0)
    v = np.full(n_paths, v0)  # v is sigma_t in SABR
    rho_bar = np.sqrt(1 - rho**2)

    for i in range(n_steps):
        z1 = dw1[:, i]
        z2 = dw2[:, i]
        correlated_z2 = rho * z1 + rho_bar * z2

        # Evolve spot price S_t directly
        s_pos = np.maximum(s, 1e-8)  # Avoid negative spot
        v_pos = np.maximum(v, 0)
        s += (r - q) * s_pos * dt + v_pos * (s_pos**beta) * z1

        # Evolve volatility sigma_t (lognormal process)
        v = v * np.exp(-0.5 * alpha**2 * dt + alpha * correlated_z2)

    return s


# Docstring for sabr_jump_kernel
"""
JIT-compiled kernel for SABR with log-normal jumps on the spot process.
"""


@numba.jit(nopython=True, fastmath=True, cache=True)
def sabr_jump_kernel(
    n_paths,
    n_steps,
    s0,
    v0,
    r,
    q,
    alpha,
    beta,
    rho,
    lambda_,
    mu_j,
    sigma_j,
    dt,
    dw1,
    dw2,
    jump_counts,
):
    s = np.full(n_paths, s0)
    v = np.full(n_paths, v0)  # v is sigma_t
    rho_bar = np.sqrt(1 - rho**2)
    compensator = lambda_ * (np.exp(mu_j + 0.5 * sigma_j**2) - 1)

    for i in range(n_steps):
        z1 = dw1[:, i]
        z2 = dw2[:, i]

        correlated_z2 = rho * z1 + rho_bar * z2

        s_pos = np.maximum(s, 1e-8)
        v_pos = np.maximum(v, 0)

        # SABR diffusion part
        s += (r - q - compensator) * s_pos * dt + v_pos * (s_pos**beta) * z1
        v = v * np.exp(-0.5 * alpha**2 * dt + alpha * correlated_z2)

        jumps_this_step = jump_counts[:, i]
        if np.any(jumps_this_step > 0):
            num_jumps = np.sum(jumps_this_step)

            # Jumps are multiplicative: S_t+ = S_t * exp(J)
            jump_multipliers = np.exp(np.random.normal(mu_j, sigma_j, num_jumps))
            jump_idx = 0
            for path_idx in range(n_paths):
                for _ in range(jumps_this_step[path_idx]):
                    s[path_idx] *= jump_multipliers[jump_idx]
                    jump_idx += 1

    return s


# Docstring for kou_kernel
"""
JIT-compiled kernel for Kou Double-Exponential Jump-Diffusion SDE.
"""


@numba.jit(nopython=True, fastmath=True, cache=True)
def kou_kernel(
    n_paths,
    n_steps,
    log_s0,
    r,
    q,
    sigma,
    lambda_,
    p_up,
    eta1,
    eta2,
    dt,
    dw,
    jump_counts,
):
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

            # Generate all jump sizes at once
            up_or_down = np.random.random(num_total_jumps)
            up_jumps = np.random.exponential(1.0 / eta1, num_total_jumps)
            down_jumps = -np.random.exponential(1.0 / eta2, num_total_jumps)
            all_jumps = np.where(up_or_down < p_up, up_jumps, down_jumps)

            # Apply jumps to the correct paths
            jump_idx = 0
            for path_idx in range(n_paths):
                num_jumps_on_path = jumps_this_step[path_idx]
                if num_jumps_on_path > 0:
                    log_s[path_idx] += np.sum(
                        all_jumps[jump_idx : jump_idx + num_jumps_on_path]
                    )
                    jump_idx += num_jumps_on_path

    return log_s


# Docstring for dupire_kernel
"""
JIT-compiled kernel for Dupire Local Volatility SDE simulation.
Note: Numba cannot compile the vol_surface_func itself, so this kernel
will have a slight overhead from calling back into Python for the vol lookup.
"""


@numba.jit(nopython=True, fastmath=True, cache=True)
def dupire_kernel(
    n_paths,
    n_steps,
    log_s0,
    r,
    q,
    dt,
    dw,
    vol_surface_func,
):
    log_s = np.full(n_paths, log_s0)
    for i in range(n_steps):
        t_current = i * dt
        current_spot = np.exp(log_s)
        local_vol = vol_surface_func(t_current, current_spot)
        drift = (r - q - 0.5 * local_vol**2) * dt
        log_s += drift + local_vol * dw[:, i]

    return log_s
