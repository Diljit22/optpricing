from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from optpricing.atoms import Option, Rate, Stock
from optpricing.models import BaseModel
from optpricing.techniques.base.pricing_result import PricingResult

__doc__ = """
Defines the abstract base class for all pricing techniques.
"""


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
        **kwargs,
    ) -> PricingResult | np.ndarray:
        """
        Calculate the price of an option.

        Parameters
        ----------
        option : Option | np.ndarray
            The option contract(s) to be priced.
        stock : Stock
            The underlying asset's properties.
        model : BaseModel
            The financial model to use for the calculation.
        rate : Rate
            The risk-free rate structure.
        **kwargs : Any
            Additional keyword arguments required by specific techniques or models.

        Returns
        -------
        PricingResult
            An object containing the calculated price and potentially other metrics.
        """
        raise NotImplementedError
