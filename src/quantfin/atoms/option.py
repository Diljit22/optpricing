from __future__ import annotations
from dataclasses import dataclass, replace
from enum import Enum
from typing import Union

class OptionType(Enum):
    CALL = "Call"
    PUT  = "Put"

class ExerciseStyle(Enum):
    EUROPEAN = "European"
    AMERICAN  = "American"

@dataclass(frozen=True)
class Option:
    """
    Immutable container for all vanilla option parameters.

    strike        : strike price
    maturity      : time to expiry (in years)
    option_type   : call vs. put
    exercise_style: european vs. american vs. ...
    """
    strike: float
    maturity: float
    option_type: OptionType
    exercise_style: ExerciseStyle = ExerciseStyle.EUROPEAN

    def __post_init__(self):
        if self.strike <= 0:
            raise ValueError(f"strike must be positive, got {self.strike}")
        if self.maturity <= 0:
            raise ValueError(f"maturity must be positive, got {self.maturity}")
    
    def parity_counterpart(self) -> Option:
        """Flip OptionType."""
        other = OptionType.CALL if self.option_type is OptionType.PUT else OptionType.PUT
        return replace(self, option_type=other)