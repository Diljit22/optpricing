from __future__ import annotations

__doc__ = """
The `kernels` package contains low-level, high-performance numerical
implementations that power the pricing techniques.

These functions are designed to be pure, operating on primitive data types,
making them ideal for JIT-compilation with `numba`.
"""

from .american_mc_kernels import longstaff_schwartz_pricer
from .lattice_kernels import (
    _crr_pricer,
    _lr_pricer,
    _topm_pricer,
)
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
from .path_kernels import (
    bates_path_kernel,
    bsm_path_kernel,
    heston_path_kernel,
    kou_path_kernel,
    merton_path_kernel,
    sabr_jump_path_kernel,
    sabr_path_kernel,
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
    # Monte Carlo Path Kernels
    "bsm_path_kernel",
    "heston_path_kernel",
    "merton_path_kernel",
    "bates_path_kernel",
    "sabr_path_kernel",
    "sabr_jump_path_kernel",
    "kou_path_kernel",
    # MC Path Pricer
    "longstaff_schwartz_pricer",
]
