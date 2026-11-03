from __future__ import annotations

from collections.abc import Callable

import numpy as np

__doc__ = """
Inverse-transform sampling of a univariate law from its characteristic function.

Used by pure-jump Levy models whose terminal distribution has no tractable exact
sampler at an arbitrary horizon (CGMY for general Y, the hyperbolic Levy motion
for t != 1). The density is recovered with the COS expansion of Fang & Oosterlee
(2008) on a cumulant-based truncation range, integrated to a CDF on a fine grid,
and inverted by interpolation.
"""


def _cumulants_from_cf(cf: Callable, h: float = 1e-3) -> tuple[float, float]:
    """First two cumulants from central differences of the CF at the origin."""
    phi_p, phi_m = cf(h), cf(-h)
    c1 = float(np.imag(phi_p - phi_m) / (2.0 * h))
    second_moment = -float(np.real(phi_p - 2.0 + phi_m) / h**2)
    c2 = max(second_moment - c1**2, 1e-16)
    return c1, c2


def cos_cdf_table(
    cf: Callable,
    *,
    n_terms: int = 4096,
    n_grid: int = 8193,
    width: float = 12.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Tabulate the CDF implied by a characteristic function on a regular grid.

    Parameters
    ----------
    cf : Callable
        Characteristic function ``u -> E[exp(iuX)]`` accepting numpy arrays.
    n_terms : int, optional
        Number of COS terms, by default 4096.
    n_grid : int, optional
        Number of grid points for the tabulated CDF, by default 8193.
    width : float, optional
        Half-width of the truncation range in standard deviations, by default 12.
        The range is widened automatically if it does not capture the mass.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(x_grid, cdf)`` with ``cdf`` non-decreasing from 0 to 1.
    """
    c1, c2 = _cumulants_from_cf(cf)
    std = np.sqrt(c2)

    for _ in range(6):
        a, b = c1 - width * std, c1 + width * std
        span = b - a
        k = np.arange(n_terms)
        omega = k * np.pi / span
        coeffs = (2.0 / span) * np.real(cf(omega) * np.exp(-1j * omega * a))
        coeffs[0] *= 0.5

        x = np.linspace(a, b, n_grid)
        # CDF(x) = int_a^x density
        #        = c_0 (x-a) + sum_{k>=1} c_k (span/(k pi)) sin(k pi (x-a)/span)
        phase = np.outer(x - a, omega[1:])
        sine_weights = coeffs[1:] * span / (k[1:] * np.pi)
        cdf = coeffs[0] * (x - a) + np.sin(phase) @ sine_weights

        total = cdf[-1]
        edge_mass = max(abs(cdf[1] - cdf[0]), abs(cdf[-1] - cdf[-2])) / (x[1] - x[0])
        if abs(total - 1.0) < 1e-6 and edge_mass * std < 1e-7:
            break
        width *= 1.5  # heavy tails: widen and retry

    cdf = np.maximum.accumulate(np.clip(cdf, 0.0, None))
    cdf /= cdf[-1]
    return x, cdf


def sample_from_cf(
    cf: Callable,
    size: int,
    rng: np.random.Generator,
    **table_kwargs,
) -> np.ndarray:
    """
    Draw ``size`` samples of the law with characteristic function ``cf``.

    Parameters
    ----------
    cf : Callable
        Characteristic function accepting numpy arrays.
    size : int
        Number of samples.
    rng : np.random.Generator
        Random number generator used for the uniforms.
    **table_kwargs
        Forwarded to :func:`cos_cdf_table`.

    Returns
    -------
    np.ndarray
        Samples obtained by inverse transform of the tabulated CDF.
    """
    x, cdf = cos_cdf_table(cf, **table_kwargs)
    u = rng.uniform(0.0, 1.0, size=size)
    return np.interp(u, cdf, x)
