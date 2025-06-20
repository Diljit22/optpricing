from __future__ import annotations
__doc__ = """
The `atoms` package provides the fundamental data structures used throughout
the quantfin library, representing core financial concepts like options,
stocks, and interest rates.
"""

from .option import Option, OptionType, ExerciseStyle
from .rate import Rate
from .stock import Stock
from .bond import ZeroCouponBond

__all__ = [
    "Option",
    "OptionType",
    "ExerciseStyle",
    "Rate",
    "Stock",
    "ZeroCouponBond",
]