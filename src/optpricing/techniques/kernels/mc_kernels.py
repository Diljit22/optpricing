from __future__ import annotations

import math

import numba
import numpy as np

__doc__ = """
JIT-compiled (`numba`) kernels that simulate the terminal value of an SDE for a
chunk of Monte Carlo paths.

Every kernel draws its own random numbers from the ``numpy.random.Generator``
it receives, so no increment arrays are materialised and the work can be split
into independent chunks (one generator each, spawned from a single seed) that
run concurrently: the kernels release the GIL. Results are reproducible for a
given seed and chunk size, independently of the number of threads.

With ``antithetic=True`` the first ``n_paths // 2`` paths use the drawn
increments and the second half their negatives; ``n_paths`` must then be even.
Jump counts are shared by an antithetic pair, jump sizes are drawn per path.
"""


@numba.njit(cache=True, nogil=True)
def bsm_kernel(gen, n_paths, n_steps, antithetic, log_s0, r, q, sigma, dt):
    """Geometric Brownian motion: d log S = (r - q - sigma^2/2) dt + sigma dW."""
    half = n_paths // 2 if antithetic else n_paths
    n_signs = 2 if antithetic else 1
    log_s = np.full(n_paths, log_s0)
    drift = (r - q - 0.5 * sigma * sigma) * dt
    vol = sigma * math.sqrt(dt)
    for _ in range(n_steps):
        z = gen.standard_normal(half)
        for s_idx in range(n_signs):
            sign = 1.0 if s_idx == 0 else -1.0
            off = s_idx * half
            for j in range(half):
                log_s[off + j] += drift + vol * sign * z[j]
    return log_s


@numba.njit(cache=True, nogil=True)
def heston_kernel(
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
    """
    Heston, full-truncation Euler (Lord, Koekkoek & van Dijk 2010): the variance
    state may go negative and only its positive part enters drift and diffusion.
    """
    half = n_paths // 2 if antithetic else n_paths
    n_signs = 2 if antithetic else 1
    log_s = np.full(n_paths, log_s0)
    v = np.full(n_paths, v0)
    rho_bar = math.sqrt(1.0 - rho * rho)
    sdt = math.sqrt(dt)
    for _ in range(n_steps):
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
    return log_s


@numba.njit(cache=True, nogil=True)
def merton_kernel(
    gen, n_paths, n_steps, antithetic, log_s0, r, q, sigma, lambda_, mu_j, sigma_j, dt
):
    """Merton jump diffusion with compensated log-normal jumps."""
    half = n_paths // 2 if antithetic else n_paths
    n_signs = 2 if antithetic else 1
    log_s = np.full(n_paths, log_s0)
    compensator = lambda_ * (math.exp(mu_j + 0.5 * sigma_j * sigma_j) - 1.0)
    drift = (r - q - 0.5 * sigma * sigma - compensator) * dt
    vol = sigma * math.sqrt(dt)
    for _ in range(n_steps):
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
    return log_s


@numba.njit(cache=True, nogil=True)
def bates_kernel(
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
    """Bates: Heston diffusion plus compensated log-normal jumps."""
    half = n_paths // 2 if antithetic else n_paths
    n_signs = 2 if antithetic else 1
    log_s = np.full(n_paths, log_s0)
    v = np.full(n_paths, v0)
    compensator = lambda_ * (math.exp(mu_j + 0.5 * sigma_j * sigma_j) - 1.0)
    rho_bar = math.sqrt(1.0 - rho * rho)
    sdt = math.sqrt(dt)
    for _ in range(n_steps):
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
    return log_s


@numba.njit(cache=True, nogil=True)
def kou_kernel(
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
    """Kou double-exponential jump diffusion with compensated jumps."""
    half = n_paths // 2 if antithetic else n_paths
    n_signs = 2 if antithetic else 1
    log_s = np.full(n_paths, log_s0)
    compensator = lambda_ * (
        p_up * eta1 / (eta1 - 1.0) + (1.0 - p_up) * eta2 / (eta2 + 1.0) - 1.0
    )
    drift = (r - q - 0.5 * sigma * sigma - compensator) * dt
    vol = sigma * math.sqrt(dt)
    for _ in range(n_steps):
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
    return log_s


@numba.njit(cache=True, nogil=True)
def _sabr_step(gen, f, v, dz, nu, beta, rho, dt):
    """
    One step of the low-bias SABR scheme of Chen, Oosterlee & van der Weide
    (2012) for the volatility Brownian increment ``dz``; returns ``(f, v)``.

    The volatility is exact, ``A = (v^2 + v_new^2) dt / 2`` is the trapezoidal
    integrated variance and ``int sigma dZ = (v_new - v) / nu``. ``Y =
    F^(1-beta) / (1-beta)`` moves by ``rho int sigma dZ - beta rho^2 A /
    (2 (1-beta) Y)`` (the part of Ito's drift the CEV draw below does not
    supply) and then by a driftless CEV with clock ``(1 - rho^2) A``, drawn
    exactly from the absorbed squared-Bessel law (Makarov & Glew 2010).
    ``beta = 0`` (normal) and ``beta = 1`` (lognormal) use their exact Gaussian
    conditionals. Zero is absorbing for 0 < beta < 1.
    """
    log_move = nu * dz - 0.5 * nu * nu * dt
    v_new = v * math.exp(log_move)
    if f <= 0.0 and beta > 0.0:
        return 0.0, v_new
    var = 0.5 * (v * v + v_new * v_new) * dt
    vol_integral = v * math.expm1(log_move) / nu if nu > 0.0 else v * dz
    shift = rho * vol_integral
    clock = (1.0 - rho * rho) * var
    if beta == 1.0:
        noise = math.sqrt(clock) * gen.standard_normal()
        return f * math.exp(shift - 0.5 * var + noise), v_new
    if beta == 0.0:
        return f + shift + math.sqrt(clock) * gen.standard_normal(), v_new
    one_m = 1.0 - beta
    y_power = f**one_m
    base = y_power + one_m * shift - 0.5 * beta * one_m * rho * rho * var / y_power
    if base <= 0.0:
        return 0.0, v_new
    if clock <= 0.0:
        return base ** (1.0 / one_m), v_new
    lam = base * base / (one_m * one_m * 2.0 * clock)
    if lam > 1e10:
        # Gaussian limit of the CEV step, relative error O(1 / lam).
        drift = 0.5 * beta * one_m * clock / base
        moved = base - drift + one_m * math.sqrt(clock) * gen.standard_normal()
        return (moved ** (1.0 / one_m) if moved > 0.0 else 0.0), v_new
    g = gen.gamma(0.5 / one_m, 1.0)
    if g >= lam:
        return 0.0, v_new
    x = gen.gamma(gen.poisson(lam - g) + 1.0, 2.0 * clock)
    return (one_m * one_m * x) ** (0.5 / one_m), v_new


@numba.njit(cache=True, nogil=True)
def sabr_kernel(gen, n_paths, n_steps, antithetic, f0, v0, nu, beta, rho, dt):
    """
    SABR forward dynamics dF = sigma_t F^beta dW, d sigma_t = nu sigma_t dZ,
    d<W,Z> = rho dt, simulated with the low-bias scheme of `_sabr_forward_step`.
    `f0` is the (shifted) initial forward; the terminal forward equals the
    terminal spot. Antithetic pairs share the negated volatility path.
    """
    half = n_paths // 2 if antithetic else n_paths
    n_signs = 2 if antithetic else 1
    f = np.full(n_paths, f0)
    v = np.full(n_paths, v0)
    sdt = math.sqrt(dt)
    for _ in range(n_steps):
        z = gen.standard_normal(half)
        for s_idx in range(n_signs):
            sign = 1.0 if s_idx == 0 else -1.0
            off = s_idx * half
            for j in range(half):
                k = off + j
                dz = sign * z[j] * sdt
                f[k], v[k] = _sabr_step(gen, f[k], v[k], dz, nu, beta, rho, dt)
    return f


@numba.njit(cache=True, nogil=True)
def sabr_jump_kernel(
    gen, n_paths, n_steps, antithetic, f0, v0, nu, beta, rho, lambda_, mu_j, sigma_j, dt
):
    """
    SABR forward dynamics with compensated log-normal jumps on the forward: the
    diffusion step of `sabr_kernel`, then the compensator and the jumps.
    """
    half = n_paths // 2 if antithetic else n_paths
    n_signs = 2 if antithetic else 1
    f = np.full(n_paths, f0)
    v = np.full(n_paths, v0)
    sdt = math.sqrt(dt)
    compensator = lambda_ * (math.exp(mu_j + 0.5 * sigma_j * sigma_j) - 1.0)
    jump_drift = math.exp(-compensator * dt)
    for _ in range(n_steps):
        z = gen.standard_normal(half)
        counts = gen.poisson(lambda_ * dt, half)
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
    return f


@numba.njit(cache=True, nogil=True)
def dupire_kernel(
    gen, n_paths, n_steps, antithetic, log_s0, r, q, dt, vol_surface_func
):
    """
    Local volatility: d log S = (r - q - sigma(t, S)^2/2) dt + sigma(t, S) dW.
    `vol_surface_func(t, s)` must be a numba-compiled scalar function.
    """
    half = n_paths // 2 if antithetic else n_paths
    n_signs = 2 if antithetic else 1
    log_s = np.full(n_paths, log_s0)
    sdt = math.sqrt(dt)
    for i in range(n_steps):
        t = i * dt
        z = gen.standard_normal(half)
        for s_idx in range(n_signs):
            sign = 1.0 if s_idx == 0 else -1.0
            off = s_idx * half
            for j in range(half):
                k = off + j
                local_vol = vol_surface_func(t, math.exp(log_s[k]))
                log_s[k] += (r - q - 0.5 * local_vol * local_vol) * dt + (
                    local_vol * sign * z[j] * sdt
                )
    return log_s


# ---------------------------------------------------------------------------
# Quadratic-Exponential scheme (Andersen 2008) for the Heston variance process
# ---------------------------------------------------------------------------
QE_PSI_C = 1.5


@numba.njit(cache=True, nogil=True, inline="always")
def _qe_constants(kappa, theta, rho, vol_of_vol, dt):
    """Pre-computed constants of the QE scheme with gamma_1 = gamma_2 = 1/2."""
    e = math.exp(-kappa * dt)
    c1 = vol_of_vol * vol_of_vol * e * (1.0 - e) / kappa
    c2 = theta * vol_of_vol * vol_of_vol * (1.0 - e) ** 2 / (2.0 * kappa)
    k1 = 0.5 * dt * (kappa * rho / vol_of_vol - 0.5) - rho / vol_of_vol
    k2 = 0.5 * dt * (kappa * rho / vol_of_vol - 0.5) + rho / vol_of_vol
    k3 = 0.5 * dt * (1.0 - rho * rho)
    k4 = k3
    k0_plain = -rho * kappa * theta * dt / vol_of_vol
    return e, c1, c2, k0_plain, k1, k2, k3, k4


@numba.njit(cache=True, nogil=True, inline="always")
def _qe_step(v_old, z_s, z_v, u, theta, e, c1, c2, k0_plain, k1, k2, k3, k4):
    """
    One QE step: returns (v_new, increment of log S excluding the (r-q) dt drift).
    Uses Andersen's martingale correction (eqs. 33-34) whenever it is defined.
    """
    m = theta + (v_old - theta) * e
    s2 = v_old * c1 + c2
    psi = s2 / (m * m)
    a_coef = k2 + 0.5 * k4
    if psi <= QE_PSI_C:
        inv = 2.0 / psi
        b2 = inv - 1.0 + math.sqrt(inv) * math.sqrt(inv - 1.0)
        a = m / (1.0 + b2)
        v_new = a * (math.sqrt(b2) + z_v) ** 2
        two_a_a = 2.0 * a_coef * a
        if two_a_a < 1.0:
            k0 = (
                -a_coef * b2 * a / (1.0 - two_a_a)
                + 0.5 * math.log(1.0 - two_a_a)
                - (k1 + 0.5 * k3) * v_old
            )
        else:
            k0 = k0_plain
    else:
        p = (psi - 1.0) / (psi + 1.0)
        beta = (1.0 - p) / m
        if u >= 1.0:
            u = 0.9999999999999999
        v_new = 0.0 if u <= p else math.log((1.0 - p) / (1.0 - u)) / beta
        if a_coef < beta:
            k0 = (
                -math.log(p + beta * (1.0 - p) / (beta - a_coef))
                - (k1 + 0.5 * k3) * v_old
            )
        else:
            k0 = k0_plain
    d_log_s = k0 + k1 * v_old + k2 * v_new + math.sqrt(k3 * v_old + k4 * v_new) * z_s
    return v_new, d_log_s


@numba.njit(cache=True, nogil=True)
def heston_qe_kernel(
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
    """Heston with the Quadratic-Exponential variance scheme (Andersen 2008)."""
    half = n_paths // 2 if antithetic else n_paths
    n_signs = 2 if antithetic else 1
    log_s = np.full(n_paths, log_s0)
    v = np.full(n_paths, v0)
    e, c1, c2, k0_plain, k1, k2, k3, k4 = _qe_constants(
        kappa, theta, rho, vol_of_vol, dt
    )
    drift = (r - q) * dt
    for _ in range(n_steps):
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
    return log_s


@numba.njit(cache=True, nogil=True)
def bates_qe_kernel(
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
    """Bates with the QE variance scheme and compensated log-normal jumps."""
    half = n_paths // 2 if antithetic else n_paths
    n_signs = 2 if antithetic else 1
    log_s = np.full(n_paths, log_s0)
    v = np.full(n_paths, v0)
    e, c1, c2, k0_plain, k1, k2, k3, k4 = _qe_constants(
        kappa, theta, rho, vol_of_vol, dt
    )
    compensator = lambda_ * (math.exp(mu_j + 0.5 * sigma_j * sigma_j) - 1.0)
    drift = (r - q - compensator) * dt
    for _ in range(n_steps):
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
    return log_s
