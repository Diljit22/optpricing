from __future__ import annotations

import math
from typing import Any

from quantfin.models.base import BaseModel, ParamValidator


class VasicekModel(BaseModel):
    """
    Vasicek (1977) mean-reverting short rate model.
    dr_t = kappa * (theta - r_t) * dt + sigma * dW_t
    """

    name: str = "Vasicek"
    has_closed_form: bool = True

    def _validate_params(self) -> None:
        p = self.params
        req = ["kappa", "theta", "sigma"]
        ParamValidator.require(p, req, model=self.name)
        ParamValidator.positive(p, ["kappa", "theta", "sigma"], model=self.name)

    def _closed_form_impl(self, *, spot: float, t: float, **_: Any) -> float:
        """
        Calculates the price of a Zero-Coupon Bond.

        Note: Re-interprets 'spot' as the initial short rate r0 and 't' as
        the bond's maturity T. Ignores other option-specific parameters.
        """
        r0, T = spot, t
        p = self.params
        kappa, theta, sigma = p["kappa"], p["theta"], p["sigma"]

        B = (1 / kappa) * (1 - math.exp(-kappa * T))
        A_log = (theta - sigma**2 / (2 * kappa**2)) * (B - T) - (
            sigma**2 / (4 * kappa)
        ) * B**2

        price = math.exp(A_log - B * r0)
        return price

    #  Abstract Method Implementations
    def _sde_impl(self, **kwargs: Any) -> Any:
        raise NotImplementedError

    def _cf_impl(self, **kwargs: Any) -> Any:
        raise NotImplementedError

    def _pde_impl(self, **kwargs: Any) -> Any:
        raise NotImplementedError
