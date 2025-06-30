from __future__ import annotations

__doc__ = """
The `parity` package provides models and utilities based on put-call parity.
"""

from .implied_rate import ImpliedRateModel
from .parity_model import ParityModel

__all__ = [
    "ParityModel",
    "ImpliedRateModel",
]
