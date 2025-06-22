from __future__ import annotations

from dataclasses import dataclass

__doc__ = """
This module defines the data structure for a zero-coupon bond.
"""


@dataclass(frozen=True, slots=True)
class ZeroCouponBond:
    """
    Represents a zero-coupon bond contract.

    Parameters
    ----------
    maturity : float
        The time to maturity of the bond, in years.
    face_value : float, optional
        The face value of the bond, paid at maturity. Defaults to 1.0.
    """

    maturity: float
    face_value: float = 1.0

    def __post_init__(self):
        if self.maturity <= 0:
            raise ValueError(f"Maturity must be positive, got {self.maturity}")
