from dataclasses import dataclass, field
from typing import Any

from quantfin.atoms.option import Option
from quantfin.atoms.rate import Rate
from quantfin.atoms.stock import Stock
from quantfin.models.base import BaseModel


@dataclass
class BenchmarkConfig:
    """A configuration for a single benchmark run."""

    name: str
    model: BaseModel
    model_params: dict[str, Any]
    techniques: list[tuple[Any, str]]  # list of (technique_instance, name_string)
    stock: Stock
    rate: Rate
    options: list[Option]
    technique_kwargs: dict[str, Any] = field(default_factory=dict)
    metrics_to_run: list[str] = field(
        default_factory=lambda: [
            "Price",
            "Delta",
            "Gamma",
            "Vega",
            "Theta",
            "Rho",
            "ImpliedVol",
        ]
    )
