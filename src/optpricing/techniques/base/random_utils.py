from __future__ import annotations

from contextlib import contextmanager

import numpy as np

__doc__ = """
Provides utility functions for random number generation in Monte Carlo methods.
"""


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
