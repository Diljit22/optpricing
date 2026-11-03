from __future__ import annotations

import math

import numba
import numpy as np

__doc__ = """
JIT-compiled Longstaff-Schwartz (2001) pricing algorithm.
"""


@numba.njit(cache=True, nogil=True, parallel=True)
def longstaff_schwartz_pricer(
    stock_paths: np.ndarray,
    K: float,
    dt: float,
    r: float,
    is_call: bool,
    degree: int,
    n_threads: int = 1,
) -> float:
    """
    Prices an American option by least-squares Monte Carlo.

    Parameters
    ----------
    stock_paths : np.ndarray
        A time-major ``(n_steps + 1, n_paths)`` array of simulated spot prices
        including ``t=0``.
    K : float
        Strike price.
    dt : float
        Time step size.
    r : float
        Risk-free rate.
    is_call : bool
        True for a call, False for a put.
    degree : int
        Polynomial degree of the regression basis (monomials in ``S/K``).
    n_threads : int, optional
        Number of chunks the paths are split into for the parallel passes
        (pass ``numba.get_num_threads()``; querying it inside the kernel would
        disable caching). Results do not depend on it.

    Returns
    -------
    float
        The estimated American option price.

    Notes
    -----
    At each exercise date the continuation value is regressed on monomials of
    the moneyness ``S/K`` over the in-the-money paths. The normal equations are
    accumulated as power sums in a single pass (no design matrix), which is
    well conditioned for ``degree <= 6`` because ``S/K = O(1)``.
    """
    n_steps_plus_1, n_paths = stock_paths.shape
    n_steps = n_steps_plus_1 - 1
    n_basis = degree + 1
    n_sums = 2 * degree + 1
    discount = math.exp(-r * dt)
    sign = 1.0 if is_call else -1.0

    n_threads = max(1, min(n_threads, n_paths))
    chunk = (n_paths + n_threads - 1) // n_threads

    cashflow = np.empty(n_paths)
    for p in numba.prange(n_paths):
        payoff = sign * (stock_paths[n_steps, p] - K)
        cashflow[p] = payoff if payoff > 0.0 else 0.0

    part_sums = np.zeros((n_threads, n_sums))
    part_moments = np.zeros((n_threads, n_basis))
    part_counts = np.zeros(n_threads, dtype=np.int64)
    gram = np.empty((n_basis, n_basis))
    moments = np.empty(n_basis)
    power_sums = np.empty(n_sums)

    for i in range(n_steps - 1, 0, -1):
        # Pass 1 (parallel over path chunks): discount and accumulate the
        # normal equations of the regression on the in-the-money paths.
        for t in numba.prange(n_threads):
            lo = t * chunk
            hi = min(n_paths, lo + chunk)
            for d in range(n_sums):
                part_sums[t, d] = 0.0
            for d in range(n_basis):
                part_moments[t, d] = 0.0
            count = 0
            for p in range(lo, hi):
                cashflow[p] *= discount
                intrinsic = sign * (stock_paths[i, p] - K)
                if intrinsic > 0.0:
                    count += 1
                    x = stock_paths[i, p] / K
                    y = cashflow[p]
                    x_pow = 1.0
                    for d in range(n_sums):
                        part_sums[t, d] += x_pow
                        if d < n_basis:
                            part_moments[t, d] += x_pow * y
                        x_pow *= x
            part_counts[t] = count

        n_itm = 0
        for t in range(n_threads):
            n_itm += part_counts[t]
        # The regression needs more points than basis functions to be meaningful.
        if n_itm <= n_basis:
            continue
        for d in range(n_sums):
            total = 0.0
            for t in range(n_threads):
                total += part_sums[t, d]
            power_sums[d] = total
        for d in range(n_basis):
            total = 0.0
            for t in range(n_threads):
                total += part_moments[t, d]
            moments[d] = total
        for a in range(n_basis):
            for b in range(n_basis):
                gram[a, b] = power_sums[a + b]
        beta = np.linalg.solve(gram, moments)

        # Pass 2 (parallel): exercise where intrinsic beats the fitted continuation.
        for p in numba.prange(n_paths):
            intrinsic = sign * (stock_paths[i, p] - K)
            if intrinsic > 0.0:
                x = stock_paths[i, p] / K
                continuation = 0.0
                x_pow = 1.0
                for d in range(n_basis):
                    continuation += beta[d] * x_pow
                    x_pow *= x
                if intrinsic > continuation:
                    cashflow[p] = intrinsic

    total = 0.0
    for p in range(n_paths):
        total += cashflow[p]
    return total / n_paths * discount
