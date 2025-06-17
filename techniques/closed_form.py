from abc import ABC
from typing import Dict, Any

from atoms.option import Option, OptionType
from atoms.stock  import Stock
from atoms.rate   import Rate
from techniques.base.base_technique import BaseTechnique, PricingResult
from techniques.base.greek_mixin import GreekMixin
from techniques.base.iv_mixin    import IVMixin
from models.base.base_model      import BaseModel

class ClosedFormTechnique(BaseTechnique, GreekMixin, IVMixin, ABC):
    """
    Generic wrapper for any model that provides .closed_form()
    and analytic Greeks as model.*_analytic().
    """

    def __init__(self, *, use_analytic_greeks: bool = True):
        self.use_analytic_greeks = use_analytic_greeks

    def price(
        self,
        option: Option,
        stock:  Stock,
        model:  BaseModel,
        rate:   Rate,
        **kwargs: Any,
    ) -> PricingResult:
        if not model.has_closed_form:
            raise TypeError(f"{model.name} has no closed-form solver.")

        # Base parameters
        base: Dict[str, Any] = {
            "spot":   stock.spot,
            "strike": option.strike,
            "r":      rate.rate,
            "q":      stock.dividend,
            "t":      option.maturity,
            "call":   (option.option_type is OptionType.CALL),
        }
        for key in model.cf_kwargs:
            if key in base:
                continue
            if hasattr(stock, key):
                base[key] = getattr(stock, key)
            elif key in kwargs:
                base[key] = kwargs[key]
            else:
                raise ValueError(f"{model.name} requires '{key}' for closed-form pricing.")

        price = model.closed_form(**base)
        return PricingResult(price=price)

    def delta(self, option, stock, model, rate, **kwargs):
        if self.use_analytic_greeks and hasattr(model, "delta_analytic"):
            return model.delta_analytic(
                spot=stock.spot, strike=option.strike,
                r=rate.rate, q=stock.dividend,
                t=option.maturity, call=(option.option_type is OptionType.CALL)
            )
        return super().delta(option, stock, model, rate, **kwargs)

    def gamma(self, option, stock, model, rate, **kwargs):
        if self.use_analytic_greeks and hasattr(model, "gamma_analytic"):
            return model.gamma_analytic(
                spot=stock.spot, strike=option.strike,
                r=rate.rate, q=stock.dividend,
                t=option.maturity
            )
        return super().gamma(option, stock, model, rate, **kwargs)

    def vega(self, option, stock, model, rate, **kwargs):
        if self.use_analytic_greeks and hasattr(model, "vega_analytic"):
            return model.vega_analytic(
                spot=stock.spot, strike=option.strike,
                r=rate.rate, q=stock.dividend,
                t=option.maturity
            )
        return super().vega(option, stock, model, rate, **kwargs)

    def theta(self, option, stock, model, rate, **kwargs):
        if self.use_analytic_greeks and hasattr(model, "theta_analytic"):
            return model.theta_analytic(
                spot=stock.spot, strike=option.strike,
                r=rate.rate, q=stock.dividend,
                t=option.maturity, call=(option.option_type is OptionType.CALL)
            )
        return super().theta(option, stock, model, rate, **kwargs)

    def rho(self, option, stock, model, rate, **kwargs):
        if self.use_analytic_greeks and hasattr(model, "rho_analytic"):
            return model.rho_analytic(
                spot=stock.spot, strike=option.strike,
                r=rate.rate, q=stock.dividend,
                t=option.maturity, call=(option.option_type is OptionType.CALL)
            )
        return super().rho(option, stock, model, rate, **kwargs)
