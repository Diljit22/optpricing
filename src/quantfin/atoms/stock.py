from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class Stock:
    """Holds stock parameters."""
    spot: float
    volatility: float | None = None
    dividend: float = 0.0
    discrete_dividends: List[float] | None = None
    ex_div_times: List[float] | None = None