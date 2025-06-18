from __future__ import annotations
import math
from typing import Any

from quantfin.models.base import BaseModel, ParamValidator



class PerpetualPutModel(BaseModel):
    """
    Closed-form price for a perpetual American put.
    """
    name = "Perpetual Put"
    supports_cf     = False
    supports_sde    = False
    supports_pde    = False
    has_closed_form = True
    supported_lattices: set[str] = set()
    cf_kwargs = BaseModel.cf_kwargs

    def __init__(self, *, params: dict[str, float]) -> None:
        super().__init__(params=params)

    def _validate_params(self) -> None:
        ParamValidator.require(self.params, ["sigma", "rate"], model=self.name)
        ParamValidator.positive(self.params, ["sigma", "rate"], model=self.name)

    def _cf_impl(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(f"{self.name} characteristic function not implemented")

    def _sde_impl(self) -> Any:
        raise NotImplementedError(f"{self.name} SDE not implemented")

    def _pde_impl(self) -> Any:
        raise NotImplementedError(f"{self.name} PDE not implemented")

    def _closed_form_impl(
        self,
        *,
        spot: float,
        strike: float,
        r: float,
        q: float,
        **_: Any
    ) -> float:
        r0    = self.params["rate"]
        sigma = self.params["sigma"]

        vol2  = sigma * sigma
        b     = r0 - q - 0.5 * vol2
        disc  = b * b + 2.0 * r0 * vol2
        beta  = -(b + math.sqrt(disc)) / vol2

        mroot = beta - 1.0
        mul   = -strike / mroot
        base  = (mroot / beta) * (spot / strike)
        return mul * (base ** beta)
