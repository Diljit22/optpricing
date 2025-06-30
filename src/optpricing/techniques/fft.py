from __future__ import annotations

import math
from typing import Any

import numpy as np

from optpricing.atoms import Option, OptionType, Rate, Stock
from optpricing.models import BaseModel
from optpricing.techniques.base import BaseTechnique, GreekMixin, IVMixin, PricingResult

__doc__ = """
Defines a pricing technique based on the Fast Fourier Transform (FFT) of the
dampened, risk-neutral option price, following the Carr-Madan formula.
"""


class FFTTechnique(BaseTechnique, GreekMixin, IVMixin):
    """
    Fast Fourier Transform (FFT) pricer based on the Carr-Madan formula,
    preserving the original tuned logic for grid and parameter selection.
    """

    def __init__(
        self,
        *,
        n: int = 12,
        eta: float = 0.25,
        alpha: float | None = None,
    ):
        """
        Initializes the FFT solver.

        Parameters
        ----------
        n : int, optional
            The exponent for the number of grid points (N = 2^n), by default 12.
        eta : float, optional
            The spacing of the grid in the frequency domain, by default 0.25.
        alpha : float | None, optional
            The dampening parameter. If None, it is auto-tuned based on a
            volatility proxy from the model. Defaults to None.
        """
        self.n = int(n)
        self.N = 1 << self.n
        self.base_eta = float(eta)
        self.alpha_user = alpha
        self._cached_results: dict[str, Any] = {}

    def _price_and_greeks(
        self,
        option: Option,
        stock: Stock,
        model: BaseModel,
        rate: Rate,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Internal method to perform the core FFT calculation once.

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

        S0, K, T = stock.spot, option.strike, option.maturity
        r, q = rate.get_rate(T), stock.dividend

        vol_proxy = self._get_vol_proxy(model, kwargs)

        if self.alpha_user is not None:
            alpha = self.alpha_user
        elif vol_proxy is None:
            alpha = 1.75
        else:
            alpha = 1.0 + 0.5 * vol_proxy * math.sqrt(T)

        eta = (
            self.base_eta * max(1.0, vol_proxy * math.sqrt(T))
            if vol_proxy is not None
            else self.base_eta
        )

        lambda_ = (2 * math.pi) / (self.N * eta)
        b = (self.N * lambda_) / 2.0
        k_grid = -b + lambda_ * np.arange(self.N)

        # Simpson's rule weights
        w = np.ones(self.N)
        w[1:-1:2], w[2:-2:2] = 4, 2
        weights = w * eta / 3.0

        phi = model.cf(t=T, spot=S0, r=r, q=q, **kwargs)

        u = np.arange(self.N) * eta
        discount = math.exp(-r * T)
        numerator = phi(u - 1j * (alpha + 1))
        denominator = alpha**2 + alpha - u**2 + 1j * u * (2 * alpha + 1)
        psi = discount * numerator / denominator

        fft_input = psi * np.exp(1j * u * b) * weights
        fft_vals = np.fft.fft(fft_input).real

        call_price_grid = np.exp(-alpha * k_grid) / math.pi * fft_vals

        # Interpolate to find the results at the target strike
        k_target = math.log(K)
        call_price = np.interp(k_target, k_grid, call_price_grid)

        if option.option_type is OptionType.CALL:
            price = call_price
        else:
            price = call_price - (S0 * np.exp(-q * T) - K * np.exp(-r * T))

        return {"price": price}

    def price(
        self,
        option: Option,
        stock: Stock,
        model: BaseModel,
        rate: Rate,
        **kwargs: Any,
    ) -> PricingResult:
        """
        Calculates the option price using the FFT method.

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
        self._cached_results = self._price_and_greeks(
            option, stock, model, rate, **kwargs
        )
        return PricingResult(price=self._cached_results["price"])

    @staticmethod
    def _get_vol_proxy(
        model: BaseModel,
        kw: dict[str, Any],
    ) -> float | None:
        """
        Finds a best-effort volatility proxy from the model or kwargs.

        This is used for the heuristic auto-tuning of the FFT grid parameters.
        It checks for 'sigma', 'v0', and 'vol_of_vol' in that order.
        """
        if "sigma" in model.params and model.params["sigma"] is not None:
            return model.params["sigma"]
        if "v0" in kw and kw["v0"] is not None:
            return math.sqrt(kw["v0"])
        if "v0" in model.params and model.params["v0"] is not None:
            return math.sqrt(model.params["v0"])
        if "vol_of_vol" in model.params and model.params["vol_of_vol"] is not None:
            return model.params["vol_of_vol"]
        return None
