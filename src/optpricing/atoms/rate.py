from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

__doc__ = """
This module defines the data structure for representing interest rates.
"""


@dataclass(frozen=True, slots=True)
class Rate:
    """
    Represents the risk-free interest rate structure.

    This can be a single constant rate or a full term structure.

    Parameters
    ----------
    rate : float | Callable[[float], float]
        - If a float, represents a constant risk-free rate for all maturities.
        - If a callable, it should be a function that takes a maturity (t)
            and returns the continuously compounded zero rate for that maturity.
            Example: `lambda t: 0.02 + 0.01 * t`
    """

    rate: float | Callable[[float], float]

    def get_rate(self, t: float = 0) -> float:
        """
        Get the interest rate for a specific maturity.

        Parameters
        ----------
        t : float, optional
            The time (maturity) for which to get the rate. This is only
            used if the rate is a term structure (callable). Default is 0.

        Returns
        -------
        float
            The continuously compounded risk-free rate.
        """
        if callable(self.rate):
            return self.rate(t)
        return self.rate
