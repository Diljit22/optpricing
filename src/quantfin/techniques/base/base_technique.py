from __future__ import annotations
from abc import ABC, abstractmethod

from quantfin.atoms import Option, Stock, Rate
from quantfin.models import BaseModel
from quantfin.techniques.base.pricing_result import PricingResult

import numpy as np

class BaseTechnique(ABC):
    """
    Abstract base class for all pricing methodologies.

    A technique defines the algorithm used to compute a price from the core
    'atoms' (Option, Stock, Rate) and a given financial 'Model'.
    """
    @abstractmethod
    def price(
        self,
        option: Option | np.ndarray,
        stock: Stock,
        model: BaseModel,
        rate: Rate,
        **kwargs
        ) -> PricingResult | np.ndarray:
        """
        Calculate the price of an option.

        Parameters
        ----------
        option : Option
            The option contract to be priced.
        stock : Stock
            The underlying asset's properties.
        model : BaseModel
            The financial model to use for the calculation.
        rate : Rate
            The risk-free rate structure.

        Returns
        -------
        PricingResult
            An object containing the calculated price and potentially other metrics.
        """
        raise NotImplementedError
    