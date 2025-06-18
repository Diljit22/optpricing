from dataclasses import replace
from typing import Any

import numpy as np

from quantfin.atoms.option                   import Option
from quantfin.atoms.stock                    import Stock
from quantfin.atoms.rate                     import Rate
from quantfin.models.base                     import BaseModel
from quantfin.techniques.base.base_technique  import PricingResult
from quantfin.techniques.base.random_utils    import crn
from contextlib import contextmanager

@contextmanager
def _crn(rng: np.random.Generator):
    """Save & restore RNG state so two pricer calls share the same path."""
    state = rng.bit_generator.state
    try:
        yield
    finally:
        rng.bit_generator.state = state

class GreekMixin:
    """
    Default finite-difference Greeks:
    Assumes `self.price(option, stock, model, rate)` -> PricingResult.
    """

    def delta(
        self,
        option: Option,
        stock:   Stock,
        model:   BaseModel,
        rate:    Rate,
        h_frac:  float = 1e-3,
        **kwargs: Any,
    ) -> float:
        S0, h = stock.spot, stock.spot * h_frac
        rng: np.random.Generator | None = getattr(self, "rng", None)

        # ---------------------  CRN branch for MC  --------------------- #
        if isinstance(rng, np.random.Generator):
            with _crn(rng):
                stock.spot = S0 + h
                p_up = self.price(option, stock, model, rate, **kwargs).price

            with _crn(rng):
                stock.spot = S0 - h
                p_dn = self.price(option, stock, model, rate, **kwargs).price

            stock.spot = S0
            return (p_up - p_dn) / (2 * h)

        # ------------------  Fallback finite diff  --------------------- #
        try:
            stock.spot = S0 + h
            p_up = self.price(option, stock, model, rate, **kwargs).price

            stock.spot = S0 - h
            p_dn = self.price(option, stock, model, rate, **kwargs).price
        finally:
            stock.spot = S0

        return (p_up - p_dn) / (2 * h)

    def gamma(
        self,
        option: Option,
        stock:  Stock,
        model:  BaseModel,
        rate:   Rate,
        h_frac: float = 1e-3,
        **kw: Any,
    ) -> float:
        S0, h = stock.spot, stock.spot * h_frac
        rng: np.random.Generator | None = getattr(self, "rng", None)

        if isinstance(rng, np.random.Generator):
            with _crn(rng):
                stock.spot = S0 + h
                p_up = self.price(option, stock, model, rate, **kw).price
            with _crn(rng):
                stock.spot = S0
                p_0 = self.price(option, stock, model, rate, **kw).price
            with _crn(rng):
                stock.spot = S0 - h
                p_dn = self.price(option, stock, model, rate, **kw).price
            stock.spot = S0
        else:
            try:
                stock.spot = S0 + h
                p_up = self.price(option, stock, model, rate, **kw).price
                stock.spot = S0
                p_0 = self.price(option, stock, model, rate, **kw).price
                stock.spot = S0 - h
                p_dn = self.price(option, stock, model, rate, **kw).price
            finally:
                stock.spot = S0

        return (p_up - 2 * p_0 + p_dn) / (h * h)

    def vega(
        self,
        option: Option,
        stock:  Stock,
        model:  BaseModel,
        rate:   Rate,
        h: float = 1e-4,
        **kw: Any,
    ) -> float:
        # Decide which volatility parameter to bump
        if "sigma" in model.params:
            get_vol = lambda: model.params["sigma"]
            set_vol = lambda v: model.params.__setitem__("sigma", v)
        elif stock.volatility is not None:
            get_vol = lambda: stock.volatility
            set_vol = lambda v: setattr(stock, "volatility", v)
        else:
            return np.nan

        sig_ = get_vol()
        rng: np.random.Generator | None = getattr(self, "rng", None)

        if isinstance(rng, np.random.Generator):
            with _crn(rng):
                set_vol(sig_ + h)
                p_up = self.price(option, stock, model, rate, **kw).price
            with _crn(rng):
                set_vol(sig_ - h)
                p_dn = self.price(option, stock, model, rate, **kw).price
            set_vol(sig_)
        else:
            try:
                set_vol(sig_ + h); p_up = self.price(option, stock, model, rate, **kw).price
                set_vol(sig_ - h); p_dn = self.price(option, stock, model, rate, **kw).price
            finally:
                set_vol(sig_)

        return (p_up - p_dn) / (2 * h)

    def theta(
        self,
        option: Option,
        stock:  Stock,
        model:  BaseModel,
        rate:   Rate,
        h: float = 1e-5,
        **kw: Any,
    ) -> float:
        T0 = option.maturity
        rng: np.random.Generator | None = getattr(self, "rng", None)

        opt_up = replace(option, maturity=T0 + h)
        opt_dn = replace(option, maturity=max(T0 - h, 0.0))

        if isinstance(rng, np.random.Generator):
            with _crn(rng):
                p_up = self.price(opt_up, stock, model, rate, **kw).price
            with _crn(rng):
                p_dn = self.price(opt_dn, stock, model, rate, **kw).price
        else:
            p_up = self.price(opt_up, stock, model, rate, **kw).price
            p_dn = self.price(opt_dn, stock, model, rate, **kw).price

        return (p_dn - p_up) / (2 * h)

    def rho(
        self,
        option: Option,
        stock:  Stock,
        model:  BaseModel,
        rate:   Rate,
        h: float = 1e-4,
        **kw: Any,
    ) -> float:
        r0 = rate.rate
        rng: np.random.Generator | None = getattr(self, "rng", None)

        if isinstance(rng, np.random.Generator):
            with _crn(rng):
                rate.rate = r0 + h
                p_up = self.price(option, stock, model, rate, **kw).price
            with _crn(rng):
                rate.rate = r0 - h
                p_dn = self.price(option, stock, model, rate, **kw).price
            rate.rate = r0
        else:
            try:
                rate.rate = r0 + h
                p_up = self.price(option, stock, model, rate, **kw).price
                rate.rate = r0 - h
                p_dn = self.price(option, stock, model, rate, **kw).price
            finally:
                rate.rate = r0

        return (p_up - p_dn) / (2 * h)
