from __future__ import annotations

from typing import Any, Dict

from quantfin.atoms import Option, OptionType, Rate, Stock, ZeroCouponBond
from quantfin.models import BaseModel
from quantfin.techniques.base import BaseTechnique, GreekMixin, IVMixin, PricingResult


class ClosedFormTechnique(BaseTechnique, GreekMixin, IVMixin):
    """
    A pricing technique for models that provide a closed-form solution.

    This class acts as a generic wrapper. It calls the `price_closed_form`
    method on a given model. It also intelligently uses analytic Greeks if the
    model provides them, otherwise falling back to the finite-difference methods
    from `GreekMixin`.
    """

    def __init__(self, *, use_analytic_greeks: bool = True):
        """
        Initializes the technique.

        Parameters
        ----------
        use_analytic_greeks : bool, optional
            If True, the technique will use the model's specific analytic Greek
            methods (e.g., `delta_analytic`) if they exist. If False or if the
            methods don't exist, it falls back to finite differences.
            Default is True.
        """
        self.use_analytic_greeks = use_analytic_greeks

    def price(self, option: Option | ZeroCouponBond, stock: Stock, model: BaseModel, rate: Rate, **kwargs: Any) -> PricingResult:
        """
        Prices the instrument using the model's closed-form solution.
        """
        if not model.has_closed_form:
            raise TypeError(f"{model.name} has no closed-form solver.")

        base_params: Dict[str, Any] = {}

        if isinstance(option, Option):
            base_params = {
                "spot": stock.spot,
                "strike": option.strike,
                "r": rate.get_rate(option.maturity),
                "q": stock.dividend,
                "t": option.maturity,
                "call": (option.option_type is OptionType.CALL),
            }
        elif isinstance(option, ZeroCouponBond):
            # For rate models, 'spot' is re-interpreted as the initial short rate r0.
            base_params = {
                "spot": stock.spot,
                "t": option.maturity,
                # passed to satisfy the model signature but are ignored.
                "strike": option.face_value,
                "r": rate.get_rate(option.maturity),
                "q": stock.dividend,
            }
        else:
            raise TypeError(f"Unsupported asset type for ClosedFormTechnique: {type(option)}")

        # Add extra model-specific kwargs
        for key in getattr(model, 'cf_kwargs', []):
            if key in base_params:
                continue
            if hasattr(stock, key):
                base_params[key] = getattr(stock, key)
            elif key in kwargs:
                base_params[key] = kwargs[key]
            else:
                if not (isinstance(option, ZeroCouponBond) and key in ['call_price', 'put_price']):
                     raise ValueError(f"{model.name} requires '{key}' for closed-form pricing.")

        # For ImpliedRateModel, pass these explicitly
        if 'call_price' in kwargs: base_params['call_price'] = kwargs['call_price']
        if 'put_price' in kwargs: base_params['put_price'] = kwargs['put_price']

        price = model.price_closed_form(**base_params)
        return PricingResult(price=price)

    def delta(self, option: Option, stock: Stock, model: BaseModel, rate: Rate, **kwargs: Any) -> float:
        """Overrides GreekMixin to use analytic delta if available."""
        if self.use_analytic_greeks and hasattr(model, "delta_analytic"):
            return model.delta_analytic(
                spot=stock.spot, strike=option.strike,
                r=rate.get_rate(option.maturity), q=stock.dividend,
                t=option.maturity, call=(option.option_type is OptionType.CALL)
            )
        return super().delta(option, stock, model, rate, **kwargs)

    def gamma(self, option: Option, stock: Stock, model: BaseModel, rate: Rate, **kwargs: Any) -> float:
        """Overrides GreekMixin to use analytic gamma if available."""
        if self.use_analytic_greeks and hasattr(model, "gamma_analytic"):
            return model.gamma_analytic(
                spot=stock.spot, strike=option.strike,
                r=rate.get_rate(option.maturity), q=stock.dividend,
                t=option.maturity
            )
        return super().gamma(option, stock, model, rate, **kwargs)

    def vega(self, option: Option, stock: Stock, model: BaseModel, rate: Rate, **kwargs: Any) -> float:
        """Overrides GreekMixin to use analytic vega if available."""
        if self.use_analytic_greeks and hasattr(model, "vega_analytic"):
            return model.vega_analytic(
                spot=stock.spot, strike=option.strike,
                r=rate.get_rate(option.maturity), q=stock.dividend,
                t=option.maturity
            )
        return super().vega(option, stock, model, rate, **kwargs)

    def theta(self, option: Option, stock: Stock, model: BaseModel, rate: Rate, **kwargs: Any) -> float:
        """Overrides GreekMixin to use analytic theta if available."""
        if self.use_analytic_greeks and hasattr(model, "theta_analytic"):
            return model.theta_analytic(
                spot=stock.spot, strike=option.strike,
                r=rate.get_rate(option.maturity), q=stock.dividend,
                t=option.maturity, call=(option.option_type is OptionType.CALL)
            )
        return super().theta(option, stock, model, rate, **kwargs)

    def rho(self, option: Option, stock: Stock, model: BaseModel, rate: Rate, **kwargs: Any) -> float:
        """Overrides GreekMixin to use analytic rho if available."""
        if self.use_analytic_greeks and hasattr(model, "rho_analytic"):
            return model.rho_analytic(
                spot=stock.spot, strike=option.strike,
                r=rate.get_rate(option.maturity), q=stock.dividend,
                t=option.maturity, call=(option.option_type is OptionType.CALL)
            )
        return super().rho(option, stock, model, rate, **kwargs)
