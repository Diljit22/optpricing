from __future__ import annotations

__doc__ = """
The `kernels` package contains low-level, high-performance numerical
implementations that power the pricing techniques.

These functions are designed to be pure, operating on primitive data types,
making them ideal for JIT-compilation with `numba`.
"""

from .lattice_kernels import _crr_pricer, _lr_pricer, _topm_pricer
from .mc_kernels import (
    bates_kernel,
    bsm_kernel,
    dupire_kernel,
    heston_kernel,
    kou_kernel,
    merton_kernel,
    sabr_jump_kernel,
    sabr_kernel,
)

__all__ = [
    # Lattice Kernels
    "_crr_pricer",
    "_lr_pricer",
    "_topm_pricer",
    # Monte Carlo Kernels
    "bsm_kernel",
    "heston_kernel",
    "merton_kernel",
    "bates_kernel",
    "sabr_kernel",
    "sabr_jump_kernel",
    "kou_kernel",
    "dupire_kernel",
]
