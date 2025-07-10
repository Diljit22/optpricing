from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

import numpy as np

from optpricing.techniques.base.random_utils import crn

__doc__ = """
Provides a mixin class with default numerical implementations for option Greeks.
"""

if TYPE_CHECKING:
    from optpricing.atoms import Option, Rate, Stock
    from optpricing.models import BaseModel


class GreekMixin:
    """
    Provides finite-difference calculations for Greeks.

    This mixin is designed to be side-effect-free. It creates modified copies
    of the input objects for shifted calculations rather than mutating them in place.
    It also supports Common Random Numbers (CRN) for variance reduction in
    Monte Carlo-based calculations by checking for a `self.rng` attribute.
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
        """
        Calculates delta using a central difference formula.

        Parameters
        ----------
        option : Option
            The option contract to be priced.
        stock : Stock
            The underlying asset's properties.
        model : BaseModel
            The financial model to use for the calculation.
        rate : Rate
            The risk-free rate structure.
        h_frac : float, optional
            The fractional step size for shifting the spot price, by default 1e-3.

        Returns
        -------
        float
            The calculated delta.
        """
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
        """
        Calculates gamma using a central difference formula.

        Parameters
        ----------
        option : Option
            The option contract to be priced.
        stock : Stock
            The underlying asset's properties.
        model : BaseModel
            The financial model to use for the calculation.
        rate : Rate
            The risk-free rate structure.
        h_frac : float, optional
            The fractional step size for shifting the spot price, by default 1e-3.

        Returns
        -------
        float
            The calculated gamma.
        """
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
        """
        Calculates vega using a central difference formula.

        Returns `np.nan` if the model does not have a 'sigma' parameter.

        Parameters
        ----------
        option : Option
            The option contract to be priced.
        stock : Stock
            The underlying asset's properties.
        model : BaseModel
            The financial model to use for the calculation.
        rate : Rate
            The risk-free rate structure.
        h : float, optional
            The absolute step size for shifting volatility, by default 1e-4.

        Returns
        -------
        float
            The calculated vega.
        """
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
        """
        Calculates theta using a central difference formula.

        Parameters
        ----------
        option : Option
            The option contract to be priced.
        stock : Stock
            The underlying asset's properties.
        model : BaseModel
            The financial model to use for the calculation.
        rate : Rate
            The risk-free rate structure.
        h : float, optional
            The absolute step size for shifting maturity, by default 1e-5.

        Returns
        -------
        float
            The calculated theta.
        """
        T0 = option.maturity
        opt_up = replace(option, maturity=T0 + h)
        opt_dn = replace(option, maturity=max(T0 - h, 1e-12))  # Avoid maturity

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
        """
        Calculates rho using a central difference formula.

        Parameters
        ----------
        option : Option
            The option contract to be priced.
        stock : Stock
            The underlying asset's properties.
        model : BaseModel
            The financial model to use for the calculation.
        rate : Rate
            The risk-free rate structure.
        h : float, optional
            The absolute step size for shifting the interest rate, by default 1e-4.

        Returns
        -------
        float
            The calculated rho.
        """
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
