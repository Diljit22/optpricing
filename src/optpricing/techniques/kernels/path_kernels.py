from __future__ import annotations

import math

import numba
import numpy as np

from .mc_kernels import _qe_constants, _qe_step, _sabr_step

__doc__ = """
JIT-compiled (`numba`) kernels that return the full matrix of simulated spot
paths in time-major layout, shape ``(n_steps + 1, n_paths)`` including ``t = 0``,
for American (Longstaff-Schwartz) pricing. Time-major storage keeps every
per-step write in the kernels and every per-date read in the backward induction
contiguous.

Random numbers are drawn inside the kernels from the ``numpy.random.Generator``
passed in (see ``mc_kernels``); with ``antithetic=True`` the second half of the
paths uses the negated increments of the first half.
"""


@numba.njit(cache=True, nogil=True)
def bsm_path_kernel(gen, n_paths, n_steps, antithetic, log_s0, r, q, sigma, dt):
    half = n_paths // 2 if antithetic else n_paths
    n_signs = 2 if antithetic else 1
    paths = np.empty((n_steps + 1, n_paths))
    paths[0, :] = math.exp(log_s0)
    log_s = np.full(n_paths, log_s0)
    drift = (r - q - 0.5 * sigma * sigma) * dt
    vol = sigma * math.sqrt(dt)
    for i in range(n_steps):
        z = gen.standard_normal(half)
        for s_idx in range(n_signs):
            sign = 1.0 if s_idx == 0 else -1.0
            off = s_idx * half
            for j in range(half):
                k = off + j
                log_s[k] += drift + vol * sign * z[j]
                paths[i + 1, k] = math.exp(log_s[k])
    return paths


@numba.njit(cache=True, nogil=True)
def heston_path_kernel(
    gen,
    n_paths,
    n_steps,
    antithetic,
    log_s0,
    v0,
    r,
    q,
    kappa,
    theta,
    rho,
    vol_of_vol,
    dt,
):
    half = n_paths // 2 if antithetic else n_paths
    n_signs = 2 if antithetic else 1
    paths = np.empty((n_steps + 1, n_paths))
    paths[0, :] = math.exp(log_s0)
    log_s = np.full(n_paths, log_s0)
    v = np.full(n_paths, v0)
    rho_bar = math.sqrt(1.0 - rho * rho)
    sdt = math.sqrt(dt)
    for i in range(n_steps):
        z1 = gen.standard_normal(half)
        z2 = gen.standard_normal(half)
        for s_idx in range(n_signs):
            sign = 1.0 if s_idx == 0 else -1.0
            off = s_idx * half
            for j in range(half):
                k = off + j
                dw_s = sign * z1[j] * sdt
                dw_v = sign * (rho * z1[j] + rho_bar * z2[j]) * sdt
                v_pos = v[k] if v[k] > 0.0 else 0.0
                sq = math.sqrt(v_pos)
                log_s[k] += (r - q - 0.5 * v_pos) * dt + sq * dw_s
                v[k] += kappa * (theta - v_pos) * dt + vol_of_vol * sq * dw_v
                paths[i + 1, k] = math.exp(log_s[k])
    return paths


@numba.njit(cache=True, nogil=True)
def merton_path_kernel(
    gen, n_paths, n_steps, antithetic, log_s0, r, q, sigma, lambda_, mu_j, sigma_j, dt
):
    half = n_paths // 2 if antithetic else n_paths
    n_signs = 2 if antithetic else 1
    paths = np.empty((n_steps + 1, n_paths))
    paths[0, :] = math.exp(log_s0)
    log_s = np.full(n_paths, log_s0)
    compensator = lambda_ * (math.exp(mu_j + 0.5 * sigma_j * sigma_j) - 1.0)
    drift = (r - q - 0.5 * sigma * sigma - compensator) * dt
    vol = sigma * math.sqrt(dt)
    for i in range(n_steps):
        z = gen.standard_normal(half)
        counts = gen.poisson(lambda_ * dt, half)
        for s_idx in range(n_signs):
            sign = 1.0 if s_idx == 0 else -1.0
            off = s_idx * half
            for j in range(half):
                k = off + j
                log_s[k] += drift + vol * sign * z[j]
                for _n in range(counts[j]):
                    log_s[k] += gen.normal(mu_j, sigma_j)
                paths[i + 1, k] = math.exp(log_s[k])
    return paths


@numba.njit(cache=True, nogil=True)
def bates_path_kernel(
    gen,
    n_paths,
    n_steps,
    antithetic,
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
):
    half = n_paths // 2 if antithetic else n_paths
    n_signs = 2 if antithetic else 1
    paths = np.empty((n_steps + 1, n_paths))
    paths[0, :] = math.exp(log_s0)
    log_s = np.full(n_paths, log_s0)
    v = np.full(n_paths, v0)
    compensator = lambda_ * (math.exp(mu_j + 0.5 * sigma_j * sigma_j) - 1.0)
    rho_bar = math.sqrt(1.0 - rho * rho)
    sdt = math.sqrt(dt)
    for i in range(n_steps):
        z1 = gen.standard_normal(half)
        z2 = gen.standard_normal(half)
        counts = gen.poisson(lambda_ * dt, half)
        for s_idx in range(n_signs):
            sign = 1.0 if s_idx == 0 else -1.0
            off = s_idx * half
            for j in range(half):
                k = off + j
                dw_s = sign * z1[j] * sdt
                dw_v = sign * (rho * z1[j] + rho_bar * z2[j]) * sdt
                v_pos = v[k] if v[k] > 0.0 else 0.0
                sq = math.sqrt(v_pos)
                log_s[k] += (r - q - 0.5 * v_pos - compensator) * dt + sq * dw_s
                v[k] += kappa * (theta - v_pos) * dt + vol_of_vol * sq * dw_v
                for _n in range(counts[j]):
                    log_s[k] += gen.normal(mu_j, sigma_j)
                paths[i + 1, k] = math.exp(log_s[k])
    return paths


@numba.njit(cache=True, nogil=True)
def kou_path_kernel(
    gen,
    n_paths,
    n_steps,
    antithetic,
    log_s0,
    r,
    q,
    sigma,
    lambda_,
    p_up,
    eta1,
    eta2,
    dt,
):
    half = n_paths // 2 if antithetic else n_paths
    n_signs = 2 if antithetic else 1
    paths = np.empty((n_steps + 1, n_paths))
    paths[0, :] = math.exp(log_s0)
    log_s = np.full(n_paths, log_s0)
    compensator = lambda_ * (
        p_up * eta1 / (eta1 - 1.0) + (1.0 - p_up) * eta2 / (eta2 + 1.0) - 1.0
    )
    drift = (r - q - 0.5 * sigma * sigma - compensator) * dt
    vol = sigma * math.sqrt(dt)
    for i in range(n_steps):
        z = gen.standard_normal(half)
        counts = gen.poisson(lambda_ * dt, half)
        for s_idx in range(n_signs):
            sign = 1.0 if s_idx == 0 else -1.0
            off = s_idx * half
            for j in range(half):
                k = off + j
                log_s[k] += drift + vol * sign * z[j]
                for _n in range(counts[j]):
                    if gen.random() < p_up:
                        log_s[k] += gen.exponential(1.0 / eta1)
                    else:
                        log_s[k] -= gen.exponential(1.0 / eta2)
                paths[i + 1, k] = math.exp(log_s[k])
    return paths


@numba.njit(cache=True, nogil=True)
def sabr_path_kernel(
    gen, n_paths, n_steps, antithetic, f0, v0, r, q, nu, beta, rho, dt
):
    """
    SABR forward dynamics returning *spot* paths: the forward F_t is simulated
    and converted with S_t = F_t exp(-(r - q)(T - t)), T = n_steps * dt. `f0` is
    the initial (shifted) forward; the caller removes the shift.
    """
    half = n_paths // 2 if antithetic else n_paths
    n_signs = 2 if antithetic else 1
    paths = np.empty((n_steps + 1, n_paths))
    horizon = n_steps * dt
    paths[0, :] = f0 * math.exp(-(r - q) * horizon)
    f = np.full(n_paths, f0)
    v = np.full(n_paths, v0)
    sdt = math.sqrt(dt)
    for i in range(n_steps):
        z = gen.standard_normal(half)
        discount = math.exp(-(r - q) * (horizon - (i + 1) * dt))
        for s_idx in range(n_signs):
            sign = 1.0 if s_idx == 0 else -1.0
            off = s_idx * half
            for j in range(half):
                k = off + j
                dz = sign * z[j] * sdt
                f[k], v[k] = _sabr_step(gen, f[k], v[k], dz, nu, beta, rho, dt)
                paths[i + 1, k] = f[k] * discount
    return paths


@numba.njit(cache=True, nogil=True)
def sabr_jump_path_kernel(
    gen,
    n_paths,
    n_steps,
    antithetic,
    f0,
    v0,
    r,
    q,
    nu,
    beta,
    rho,
    lambda_,
    mu_j,
    sigma_j,
    dt,
):
    """SABR forward dynamics with compensated log-normal jumps, returning spot
    paths (see `sabr_path_kernel`)."""
    half = n_paths // 2 if antithetic else n_paths
    n_signs = 2 if antithetic else 1
    paths = np.empty((n_steps + 1, n_paths))
    horizon = n_steps * dt
    paths[0, :] = f0 * math.exp(-(r - q) * horizon)
    f = np.full(n_paths, f0)
    v = np.full(n_paths, v0)
    sdt = math.sqrt(dt)
    compensator = lambda_ * (math.exp(mu_j + 0.5 * sigma_j * sigma_j) - 1.0)
    jump_drift = math.exp(-compensator * dt)
    for i in range(n_steps):
        z = gen.standard_normal(half)
        counts = gen.poisson(lambda_ * dt, half)
        discount = math.exp(-(r - q) * (horizon - (i + 1) * dt))
        for s_idx in range(n_signs):
            sign = 1.0 if s_idx == 0 else -1.0
            off = s_idx * half
            for j in range(half):
                k = off + j
                dz = sign * z[j] * sdt
                f_new, v[k] = _sabr_step(gen, f[k], v[k], dz, nu, beta, rho, dt)
                f_new *= jump_drift
                for _n in range(counts[j]):
                    f_new *= math.exp(gen.normal(mu_j, sigma_j))
                f[k] = f_new
                paths[i + 1, k] = f_new * discount
    return paths


@numba.njit(cache=True, nogil=True)
def heston_qe_path_kernel(
    gen,
    n_paths,
    n_steps,
    antithetic,
    log_s0,
    v0,
    r,
    q,
    kappa,
    theta,
    rho,
    vol_of_vol,
    dt,
):
    """Heston paths with the Quadratic-Exponential variance scheme."""
    half = n_paths // 2 if antithetic else n_paths
    n_signs = 2 if antithetic else 1
    paths = np.empty((n_steps + 1, n_paths))
    paths[0, :] = math.exp(log_s0)
    log_s = np.full(n_paths, log_s0)
    v = np.full(n_paths, v0)
    e, c1, c2, k0_plain, k1, k2, k3, k4 = _qe_constants(
        kappa, theta, rho, vol_of_vol, dt
    )
    drift = (r - q) * dt
    for i in range(n_steps):
        z_s = gen.standard_normal(half)
        z_v = gen.standard_normal(half)
        u = gen.random(half)
        for s_idx in range(n_signs):
            sign = 1.0 if s_idx == 0 else -1.0
            off = s_idx * half
            for j in range(half):
                k = off + j
                uu = u[j] if s_idx == 0 else 1.0 - u[j]
                v_new, d_log = _qe_step(
                    v[k],
                    sign * z_s[j],
                    sign * z_v[j],
                    uu,
                    theta,
                    e,
                    c1,
                    c2,
                    k0_plain,
                    k1,
                    k2,
                    k3,
                    k4,
                )
                log_s[k] += drift + d_log
                v[k] = v_new
                paths[i + 1, k] = math.exp(log_s[k])
    return paths


@numba.njit(cache=True, nogil=True)
def bates_qe_path_kernel(
    gen,
    n_paths,
    n_steps,
    antithetic,
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
):
    """Bates paths with the QE variance scheme and compensated jumps."""
    half = n_paths // 2 if antithetic else n_paths
    n_signs = 2 if antithetic else 1
    paths = np.empty((n_steps + 1, n_paths))
    paths[0, :] = math.exp(log_s0)
    log_s = np.full(n_paths, log_s0)
    v = np.full(n_paths, v0)
    e, c1, c2, k0_plain, k1, k2, k3, k4 = _qe_constants(
        kappa, theta, rho, vol_of_vol, dt
    )
    compensator = lambda_ * (math.exp(mu_j + 0.5 * sigma_j * sigma_j) - 1.0)
    drift = (r - q - compensator) * dt
    for i in range(n_steps):
        z_s = gen.standard_normal(half)
        z_v = gen.standard_normal(half)
        u = gen.random(half)
        counts = gen.poisson(lambda_ * dt, half)
        for s_idx in range(n_signs):
            sign = 1.0 if s_idx == 0 else -1.0
            off = s_idx * half
            for j in range(half):
                k = off + j
                uu = u[j] if s_idx == 0 else 1.0 - u[j]
                v_new, d_log = _qe_step(
                    v[k],
                    sign * z_s[j],
                    sign * z_v[j],
                    uu,
                    theta,
                    e,
                    c1,
                    c2,
                    k0_plain,
                    k1,
                    k2,
                    k3,
                    k4,
                )
                log_s[k] += drift + d_log
                v[k] = v_new
                for _n in range(counts[j]):
                    log_s[k] += gen.normal(mu_j, sigma_j)
                paths[i + 1, k] = math.exp(log_s[k])
    return paths
