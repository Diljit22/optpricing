from __future__ import annotations

import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import Any

import numpy as np

__doc__ = """
Random-number utilities for the Monte Carlo engines: common random numbers for
finite-difference Greeks, and chunked, multi-threaded execution of the
generator-driven numba kernels.
"""

DEFAULT_CHUNK_SIZE = 8192
_executor: ThreadPoolExecutor | None = None


@contextmanager
def crn(rng: np.random.Generator):
    """
    Context manager for Common Random Numbers (CRN).

    Saves and restores the state of a NumPy random number generator. This ensures
    that code blocks within the same `with crn(rng):` sequence will reuse the
    same random numbers, which is crucial for variance reduction when calculating
    Greeks via finite differences in Monte Carlo methods.

    Parameters
    ----------
    rng : np.random.Generator
        The random number generator instance whose state will be managed.
    """
    state = rng.bit_generator.state
    try:
        yield
    finally:
        rng.bit_generator.state = state


def default_n_jobs() -> int:
    """Number of worker threads used for Monte Carlo chunks (``OPTPRICING_MC_THREADS``
    or the CPU count)."""
    env = os.environ.get("OPTPRICING_MC_THREADS")
    if env:
        return max(1, int(env))
    return max(1, os.cpu_count() or 1)


def _get_executor() -> ThreadPoolExecutor:
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(
            max_workers=default_n_jobs(), thread_name_prefix="optpricing-mc"
        )
    return _executor


def plan_chunks(
    rng: np.random.Generator,
    n_paths: int,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    antithetic: bool = False,
) -> tuple[list[np.random.Generator], list[int]]:
    """
    Split ``n_paths`` into chunks and spawn one independent generator per chunk.

    A single 64-bit root seed is drawn from ``rng`` (so `crn` reproduces the
    same chunks) and expanded with ``numpy.random.SeedSequence.spawn``; the
    resulting streams are therefore independent of how many threads run them.

    Parameters
    ----------
    rng : np.random.Generator
        The technique's generator; advanced by exactly one draw.
    n_paths : int
        Total number of paths (made even if ``antithetic``).
    chunk_size : int, optional
        Paths per chunk, by default 8192 (even).
    antithetic : bool, optional
        If True every chunk holds an even number of paths.

    Returns
    -------
    tuple[list[np.random.Generator], list[int]]
        Per-chunk generators and per-chunk path counts (summing to ``n_paths``).
    """
    if antithetic:
        n_paths += n_paths % 2
        chunk_size += chunk_size % 2
    n_chunks = max(1, -(-n_paths // chunk_size))
    sizes = [chunk_size] * (n_chunks - 1) + [n_paths - chunk_size * (n_chunks - 1)]
    root = int(rng.integers(0, 2**63 - 1, dtype=np.int64))
    seeds = np.random.SeedSequence(root).spawn(n_chunks)
    return [np.random.default_rng(s) for s in seeds], sizes


def run_chunks(
    kernel: Callable[..., np.ndarray],
    generators: list[np.random.Generator],
    sizes: list[int],
    *,
    n_jobs: int | None = None,
    axis: int = 0,
    **kernel_kwargs: Any,
) -> np.ndarray:
    """
    Run ``kernel(gen, n_paths, **kernel_kwargs)`` for every chunk and concatenate.

    Chunks execute concurrently on a shared thread pool when more than one chunk
    exists and ``n_jobs`` allows it; the kernels must release the GIL
    (``numba.njit(nogil=True)``).

    Parameters
    ----------
    kernel : Callable
        A numba kernel whose first two positional arguments are the generator
        and the number of paths of the chunk.
    generators, sizes : list
        Output of :func:`plan_chunks`.
    n_jobs : int | None, optional
        Maximum number of concurrent chunks; ``None`` uses :func:`default_n_jobs`.
    axis : int, optional
        Axis along which the per-chunk results are concatenated (0 for terminal
        values, 1 for time-major path matrices).
    **kernel_kwargs
        Remaining keyword arguments of the kernel.

    Returns
    -------
    np.ndarray
        The per-chunk results concatenated along ``axis``.
    """
    if n_jobs is None:
        n_jobs = default_n_jobs()
    if len(generators) == 1 or n_jobs <= 1:
        parts = [kernel(g, n, **kernel_kwargs) for g, n in zip(generators, sizes)]
    else:
        executor = _get_executor()
        parts = list(
            executor.map(
                lambda gn: kernel(gn[0], gn[1], **kernel_kwargs),
                zip(generators, sizes),
            )
        )
    return np.concatenate(parts, axis=axis)
