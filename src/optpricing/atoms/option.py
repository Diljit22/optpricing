from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum

__doc__ = """
This module defines the core data structures for representing financial options.
"""


class OptionType(Enum):
    """Enumeration for the type of an option (Call or Put)."""

    CALL = "Call"
    PUT = "Put"


class ExerciseStyle(Enum):
    """Enumeration for the exercise style of an option."""

    EUROPEAN = "European"
    AMERICAN = "American"
    BERMUDAN = "Bermudan"


@dataclass(frozen=True, slots=True)
class Option:
    """
    Immutable container representing a single vanilla option contract.

    Parameters
    ----------
    strike : float
        The strike price of the option.
    maturity : float
        The time to maturity of the option, expressed in years.
    option_type : OptionType
        The type of the option, either CALL or PUT.
    exercise_style : ExerciseStyle, optional
        The exercise style of the option, by default ExerciseStyle.EUROPEAN.
    """

    strike: float
    maturity: float
    option_type: OptionType
    exercise_style: ExerciseStyle = ExerciseStyle.EUROPEAN

    def __post_init__(self):
        if self.strike <= 0:
            raise ValueError(f"Strike must be positive, got {self.strike}")
        if self.maturity <= 0:
            raise ValueError(f"Maturity must be positive, got {self.maturity}")

    def parity_counterpart(self) -> Option:
        """
        Create the put-call parity equivalent of this option.

        A call is converted to a put, and a put is converted to a call,
        while keeping all other parameters the same.

        Returns
        -------
        Option
            A new Option instance with the opposite type.
        """
        if self.option_type is OptionType.PUT:
            other_type = OptionType.CALL
        else:
            other_type = OptionType.PUT
        return replace(self, option_type=other_type)
