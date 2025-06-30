from __future__ import annotations

from typing import Any

from optpricing.atoms import Option, OptionType, Rate, Stock
from optpricing.models import BaseModel
from optpricing.techniques.base import LatticeTechnique

from .kernels.lattice_kernels import _lr_pricer

__doc__ = """
Defines the Leisen-Reimer binomial lattice pricing technique.
"""


class LeisenReimerTechnique(LatticeTechnique):
    """Leisen-Reimer binomial lattice technique with Peizer-Pratt inversion."""

    def _price_and_get_nodes(
        self,
        option: Option,
        stock: Stock,
        model: BaseModel,
        rate: Rate,
    ) -> dict[str, Any]:
        """
        Prices the option using the Leisen-Reimer kernel.

        This method extracts the necessary numerical parameters from the input
        objects and passes them to the low-level `_lr_pricer` kernel.

        Parameters
        ----------
        option : Option
            The option contract to be priced.
        stock : Stock
            The underlying asset's properties.
        model : BaseModel
            The financial model, used to get volatility.
        rate : Rate
            The risk-free rate structure.

        Returns
        -------
        dict[str, Any]
            A dictionary from the kernel containing the price and node values.
        """
        sigma = model.params.get("sigma", stock.volatility)
        return _lr_pricer(
            S0=stock.spot,
            K=option.strike,
            T=option.maturity,
            r=rate.get_rate(option.maturity),
            q=stock.dividend,
            sigma=sigma,
            N=self.steps,
            is_call=(option.option_type is OptionType.CALL),
            is_am=self.is_american,
        )
