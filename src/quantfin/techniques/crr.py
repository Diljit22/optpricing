from __future__ import annotations

from typing import Any

from quantfin.atoms import Option, OptionType, Rate, Stock
from quantfin.models import BaseModel
from quantfin.techniques.base import LatticeTechnique

from .kernels.lattice_kernels import _crr_pricer


class CRRTechnique(LatticeTechnique):
    """Cox-Ross-Rubinstein binomial lattice technique."""

    def _price_and_get_nodes(
        self,
        option: Option,
        stock: Stock,
        model: BaseModel,
        rate: Rate,
    ) -> dict[str, Any]:
        sigma = model.params.get("sigma", stock.volatility)
        return _crr_pricer(
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
