from dataclasses import dataclass, field
from typing import List, Any, Dict
from atoms.option import Option
from atoms.stock import Stock
from atoms.rate import Rate
from models.base.base_model import BaseModel

@dataclass
class BenchmarkConfig:
    """A configuration for a single benchmark run."""
    name: str
    model: BaseModel
    model_params: Dict[str, Any]
    techniques: List[tuple[Any, str]] # List of (technique_instance, name_string)
    stock: Stock
    rate: Rate
    options: List[Option]
    technique_kwargs: Dict[str, Any] = field(default_factory=dict)
    metrics_to_run: List[str] = field(default_factory=lambda: [
        "Price", "Delta", "Gamma", "Vega", "Theta", "Rho", "ImpliedVol"
    ])