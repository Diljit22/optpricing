from __future__ import annotations

from dataclasses import dataclass

__doc__ = """
This module defines the data structure for representing the underlying asset.
"""


@dataclass(frozen=True, slots=True)
class Stock:
    """
    Immutable container representing the underlying asset's properties.

    Parameters
    ----------
    spot : float
        The current price of the underlying asset.
    volatility : float | None, optional
        The constant (implied or historical) volatility of the asset's returns.
        Default is `None`.
    dividend : float, optional
        The continuously compounded dividend yield, by default 0.0.
    discrete_dividends : list[float] | None, optional
        A list of discrete dividend amounts. Default is `None`.
    ex_div_times : list[float] | None, optional
        A list of times (in years) for the discrete dividend payments.
        Must correspond to `discrete_dividends`. Default is `None`.
    """

    spot: float
    volatility: float | None = None
    dividend: float = 0.0
    discrete_dividends: list[float] | None = None
    ex_div_times: list[float] | None = None

    def __post_init__(self):
        if self.spot <= 0:
            raise ValueError(f"Spot price must be positive, got {self.spot}")
