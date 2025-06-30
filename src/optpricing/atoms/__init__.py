from __future__ import annotations

__doc__ = """
The `atoms` package provides the fundamental data structures used throughout
the optpricing library, representing core financial concepts like options,
stocks, and interest rates.
"""

from .bond import ZeroCouponBond
from .option import ExerciseStyle, Option, OptionType
from .rate import Rate
from .stock import Stock

__all__ = [
    "Option",
    "OptionType",
    "ExerciseStyle",
    "Rate",
    "Stock",
    "ZeroCouponBond",
]
