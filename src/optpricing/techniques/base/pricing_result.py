from __future__ import annotations

from dataclasses import dataclass, field

__doc__ = """
Defines a simple data container for pricing results.
"""


@dataclass(frozen=True)
class PricingResult:
    """
    A container for the results of a pricing operation.

    Attributes
    ----------
    price : float
        The calculated price of the instrument.
    greeks : dict[str, float], optional
        A dictionary containing calculated Greek values (e.g., 'delta', 'gamma').
    implied_vol : float, optional
        The calculated Black-Scholes implied volatility.
    """

    price: float
    greeks: dict[str, float] = field(default_factory=dict)
    implied_vol: float | None = None
