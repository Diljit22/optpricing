from __future__ import annotations

import math
from typing import Any

from scipy.optimize import brentq

from models.base.base_model import BaseModel
from models.base.validators import ParamValidator


class ImpliedRateModel(BaseModel):
    """Risk-free rate implied by put-call parity for European options."""

    name = "Implied Rate"
    supports_cf = False
    supports_sde = False
    supports_pde = False
    has_closed_form = True
    supported_lattices: set[str] = set()
    cf_kwargs = BaseModel.cf_kwargs + ("call_price", "put_price")


    def _validate_params(self) -> None:
        ParamValidator.require(self.params, ["eps", "max_iter"], model=self.name)
        ParamValidator.positive(self.params, ["eps", "max_iter"], model=self.name)

    def _closed_form_impl(
        self,
        *,
        call_price: float,
        put_price: float,
        spot: float,
        strike: float,
        t: float,
        q: float = 0.0,
    ) -> float:
        eps = float(self.params["eps"])
        max_iter = int(self.params["max_iter"])

        lhs = spot * math.exp(-q * t)  # discounted underlying
        diff = call_price - put_price  # C - P

        def f(r: float) -> float:
            return lhs - strike * math.exp(-r * t) - diff

        low, high = -1.0, 1.0
        fl, fh = f(low), f(high)
        cnt = 0
        while fl * fh > 0 and cnt < max_iter:
            if abs(fl) < abs(fh):
                low -= 0.5
                fl = f(low)
            else:
                high += 0.5
                fh = f(high)
            cnt += 1
        if fl * fh > 0:
            raise ValueError("Unable to bracket implied rate.")

        return brentq(f, low, high, xtol=eps, maxiter=max_iter)

    def _cf_impl(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(f"{self.name} characteristic function not implemented")

    def _sde_impl(self) -> Any:
        raise NotImplementedError(f"{self.name} SDE not implemented")

    def _pde_impl(self) -> Any:
        raise NotImplementedError(f"{self.name} PDE not implemented")