from __future__ import annotations

from typing import Any

import numpy as np
from scipy import integrate

from optpricing.atoms import Option, OptionType, Rate, Stock
from optpricing.models import BaseModel
from optpricing.techniques.base import BaseTechnique, GreekMixin, IVMixin, PricingResult

__doc__ = """
Defines a pricing technique based on the Gil-Pelaez inversion formula,
solved using numerical quadrature.
"""


class IntegrationTechnique(BaseTechnique, GreekMixin, IVMixin):
    """
    Prices options using the Gil-Pelaez inversion formula via numerical quadrature.

    This technique leverages the model's characteristic function (CF) to price
    options. It is particularly useful for models where a closed-form solution
    is unavailable but the CF is known (e.g., Heston, Bates, VG, NIG).

    It provides an analytic delta as a "free" byproduct of the pricing calculation.
    """

    def __init__(
        self,
        *,
        upper_bound: float = 200.0,
        limit: int = 200,
        epsabs: float = 1e-9,
        epsrel: float = 1e-9,
    ):
        """
        Initializes the numerical integration solver.

        Parameters
        ----------
        upper_bound : float, optional
            The upper limit of the integration, by default 200.0.
        limit : int, optional
            The maximum number of sub-intervals for the integration, by default 200.
        epsabs : float, optional
            The absolute error tolerance for the integration, by default 1e-9.
        epsrel : float, optional
            The relative error tolerance for the integration, by default 1e-9.
        """
        self.upper_bound = upper_bound
        self.limit = limit
        self.epsabs = epsabs
        self.epsrel = epsrel
        self._cached_results: dict[str, Any] = {}

    def _price_and_delta(
        self,
        option: Option,
        stock: Stock,
        model: BaseModel,
        rate: Rate,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Internal method to perform the core calculation once.

        Parameters
        ----------
        option : Option
            The option contract to be priced.
        stock : Stock
            The underlying asset's properties.
        model : BaseModel
            The financial model to use. Must support a characteristic function.
        rate : Rate
            The risk-free rate structure.
        """
        if not model.supports_cf:
            raise TypeError(
                f"Model '{model.name}' does not support a characteristic function."
            )

        S, K, T = stock.spot, option.strike, option.maturity
        r, q = rate.get_rate(T), stock.dividend
        phi = model.cf(t=T, spot=S, r=r, q=q, **kwargs)
        k_log = np.log(K)

        def integrand_p2(u):
            """Calculates the integrand for P2."""
            return (np.exp(-1j * u * k_log) * phi(u)).imag / u

        def integrand_p1(u):
            """Calculates the integrand for P1."""
            return (np.exp(-1j * u * k_log) * phi(u - 1j)).imag / u

        integral_p2, _ = integrate.quad(
            integrand_p2,
            1e-15,
            self.upper_bound,
            limit=self.limit,
            epsabs=self.epsabs,
            epsrel=self.epsrel,
        )

        phi_minus_i = phi(-1j)
        if np.abs(phi_minus_i) < 1e-12:
            P1 = np.nan
        else:
            integral_p1, _ = integrate.quad(
                integrand_p1,
                1e-15,
                self.upper_bound,
                limit=self.limit,
                epsabs=self.epsabs,
                epsrel=self.epsrel,
            )
            P1 = 0.5 + integral_p1 / (
                np.pi * np.real(phi_minus_i)
            )  # Use real part of denominator

        P2 = 0.5 + integral_p2 / np.pi

        if np.isnan(P1):
            price, delta = np.nan, np.nan
        else:
            df_S = S * np.exp(-q * T)
            df_K = K * np.exp(-r * T)
            if option.option_type is OptionType.CALL:
                price = df_S * P1 - df_K * P2
                delta = np.exp(-q * T) * P1
            else:
                price = df_K * (1 - P2) - df_S * (1 - P1)
                delta = np.exp(-q * T) * (P1 - 1.0)

        return {"price": price, "delta": delta}

    def price(
        self,
        option: Option,
        stock: Stock,
        model: BaseModel,
        rate: Rate,
        **kwargs: Any,
    ) -> PricingResult:
        """
        Calculates the option price and caches the 'free' analytic delta.

        Parameters
        ----------
        option : Option
            The option contract to be priced.
        stock : Stock
            The underlying asset's properties.
        model : BaseModel
            The financial model to use. Must support a characteristic function.
        rate : Rate
            The risk-free rate structure.

        Returns
        -------
        PricingResult
            An object containing the calculated price.
        """
        self._cached_results = self._price_and_delta(
            option, stock, model, rate, **kwargs
        )
        return PricingResult(price=self._cached_results["price"])

    def delta(
        self,
        option: Option,
        stock: Stock,
        model: BaseModel,
        rate: Rate,
        **kwargs: Any,
    ) -> float:
        """
        Returns the 'free' delta calculated during the pricing call.

        If the cache is empty or the analytic delta calculation failed, it
        falls back to the numerical finite difference method from `GreekMixin`.

        Parameters
        ----------
        option : Option
            The option contract to be priced.
        stock : Stock
            The underlying asset's properties.
        model : BaseModel
            The financial model to use. Must support a characteristic function.
        rate : Rate
            The risk-free rate structure.

        Returns
        -------
        float
            Delta of the option.
        """
        if not self._cached_results:
            self.price(option, stock, model, rate, **kwargs)

        delta_val = self._cached_results.get("delta")
        if delta_val is not None and not np.isnan(delta_val):
            return delta_val
        else:
            # Fallback to finite difference if analytic delta failed
            return super().delta(option, stock, model, rate, **kwargs)
