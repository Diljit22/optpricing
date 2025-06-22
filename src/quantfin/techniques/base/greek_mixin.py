from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

import numpy as np

from quantfin.techniques.base.random_utils import crn

if TYPE_CHECKING:
    from quantfin.atoms import Option, Rate, Stock
    from quantfin.models import BaseModel


class GreekMixin:
    """
    Provides finite-difference calculations for Greeks.

    This mixin is designed to be side-effect-free. It creates modified copies
    of the input objects for shifted calculations rather than mutating them in place.
    """

    def delta(
        self,
        option: Option,
        stock: Stock,
        model: BaseModel,
        rate: Rate,
        h_frac: float = 1e-3,
        **kwargs: Any,
    ) -> float:
        h = stock.spot * h_frac
        stock_up = replace(stock, spot=stock.spot + h)
        stock_dn = replace(stock, spot=stock.spot - h)

        rng = getattr(self, "rng", None)
        if isinstance(rng, np.random.Generator):
            with crn(rng):
                p_up = self.price(option, stock_up, model, rate, **kwargs).price
            with crn(rng):
                p_dn = self.price(option, stock_dn, model, rate, **kwargs).price
        else:
            p_up = self.price(option, stock_up, model, rate, **kwargs).price
            p_dn = self.price(option, stock_dn, model, rate, **kwargs).price

        return (p_up - p_dn) / (2 * h)

    def gamma(
        self,
        option: Option,
        stock: Stock,
        model: BaseModel,
        rate: Rate,
        h_frac: float = 1e-3,
        **kw: Any,
    ) -> float:
        h = stock.spot * h_frac
        stock_up = replace(stock, spot=stock.spot + h)
        stock_dn = replace(stock, spot=stock.spot - h)

        rng = getattr(self, "rng", None)
        if isinstance(rng, np.random.Generator):
            with crn(rng):
                p_up = self.price(option, stock_up, model, rate, **kw).price
            with crn(rng):
                p_0 = self.price(option, stock, model, rate, **kw).price
            with crn(rng):
                p_dn = self.price(option, stock_dn, model, rate, **kw).price
        else:
            p_up = self.price(option, stock_up, model, rate, **kw).price
            p_0 = self.price(option, stock, model, rate, **kw).price
            p_dn = self.price(option, stock_dn, model, rate, **kw).price

        return (p_up - 2 * p_0 + p_dn) / (h * h)

    def vega(
        self,
        option: Option,
        stock: Stock,
        model: BaseModel,
        rate: Rate,
        h: float = 1e-4,
        **kw: Any,
    ) -> float:
        if "sigma" not in model.params:
            return np.nan

        sigma = model.params["sigma"]
        model_up = model.with_params(sigma=sigma + h)
        model_dn = model.with_params(sigma=sigma - h)

        rng = getattr(self, "rng", None)
        if isinstance(rng, np.random.Generator):
            with crn(rng):
                p_up = self.price(option, stock, model_up, rate, **kw).price
            with crn(rng):
                p_dn = self.price(option, stock, model_dn, rate, **kw).price
        else:
            p_up = self.price(option, stock, model_up, rate, **kw).price
            p_dn = self.price(option, stock, model_dn, rate, **kw).price

        return (p_up - p_dn) / (2 * h)

    def theta(
        self,
        option: Option,
        stock: Stock,
        model: BaseModel,
        rate: Rate,
        h: float = 1e-5,
        **kw: Any,
    ) -> float:
        T0 = option.maturity
        opt_up = replace(option, maturity=T0 + h)
        opt_dn = replace(option, maturity=max(T0 - h, 1e-12))  # Avoid maturity of 0

        rng = getattr(self, "rng", None)
        if isinstance(rng, np.random.Generator):
            with crn(rng):
                p_up = self.price(opt_up, stock, model, rate, **kw).price
            with crn(rng):
                p_dn = self.price(opt_dn, stock, model, rate, **kw).price
        else:
            p_up = self.price(opt_up, stock, model, rate, **kw).price
            p_dn = self.price(opt_dn, stock, model, rate, **kw).price

        return (p_dn - p_up) / (2 * h)

    def rho(
        self,
        option: Option,
        stock: Stock,
        model: BaseModel,
        rate: Rate,
        h: float = 1e-4,
        **kw: Any,
    ) -> float:
        r0 = rate.get_rate(option.maturity)
        rate_up = replace(rate, rate=r0 + h)
        rate_dn = replace(rate, rate=r0 - h)

        rng = getattr(self, "rng", None)
        if isinstance(rng, np.random.Generator):
            with crn(rng):
                p_up = self.price(option, stock, model, rate_up, **kw).price
            with crn(rng):
                p_dn = self.price(option, stock, model, rate_dn, **kw).price
        else:
            p_up = self.price(option, stock, model, rate_up, **kw).price
            p_dn = self.price(option, stock, model, rate_dn, **kw).price

        return (p_up - p_dn) / (2 * h)
