from __future__ import annotations

import math
from typing import Any

import numpy as np

from quantfin.atoms.option import Option, OptionType
from quantfin.atoms.rate import Rate
from quantfin.atoms.stock import Stock
from quantfin.models.base import BaseModel
from quantfin.techniques.base.base_technique import BaseTechnique, PricingResult
from quantfin.techniques.base.greek_mixin import GreekMixin
from quantfin.techniques.base.iv_mixin import IVMixin


class LewisFFTTechnique(BaseTechnique, GreekMixin, IVMixin):
    """
    Prices European options via Lewis (2001) FFT.

    Designed for heavy-tailed models (e.g., Variance Gamma), where Carr-Madan
    can be unstable due to damping parameter sensitivity. Uses the Fourier
    transform of the option price with a shift to improve numerical stability.

    """

    def __init__(self, *, n: int = 12, eta: float = 0.25) -> None:
        """
        Initialize Lewis FFT pricer.

        Parameters
        ----------
        n : int, optional
            Log2 of FFT grid size (N = 2^n, default: 12 → N = 4096).
        eta : float, optional
            Fourier grid spacing (default: 0.25).
        """
        super().__init__()
        self.n = int(n)
        self.N = 1 << self.n
        self.base_eta = float(eta)

    def price(
        self,
        option: Option,
        stock: Stock,
        model: BaseModel,
        rate: Rate,
        **kwargs: Any,
    ) -> PricingResult:
        """
        Price an option using Lewis (2001) FFT.

        Parameters
        ----------
        option : Option
            Option details (strike, maturity, type).
        stock : Stock
            Underlying stock (spot, dividend).
        model : BaseModel
            Pricing model with characteristic function.
        rate : Rate
            Risk-free rate.
        **kwargs : Any
            Model-specific parameters (e.g., v0 for Heston).

        Returns
        -------
        PricingResult
            Option price and Greeks (if computed).

        Raises
        ------
        TypeError
            If model does not support a characteristic function.
        ValueError
            If input parameters are invalid (e.g., T <= 0).
        """
        if not model.supports_cf:
            raise TypeError(
                f"Model '{model.name}' does not support a characteristic function."
            )

        S, K, T = stock.spot, option.strike, option.maturity
        r, q = rate.rate, stock.dividend
        is_call = option.option_type is OptionType.CALL

        # Validate inputs
        if T <= 0 or S <= 0 or K <= 0:
            raise ValueError("T, S, and K must be positive.")

        # Adaptive eta based on volatility proxy
        vol_proxy = self._vol_proxy(model, kwargs)
        eta = self.base_eta * max(1.0, vol_proxy * math.sqrt(T))
        lambd = 2 * math.pi / (self.N * eta)
        b = 0.5 * self.N * lambd

        # Log-strike grid centered around log(K)
        k_grid = np.log(K) + (np.arange(self.N) - self.N / 2) * lambd

        # Build CF arguments
        cf_params: dict[str, Any] = {"t": T, "spot": S, "r": r, "q": q}
        for key in getattr(model, "cf_kwargs", ()):
            if key in kwargs:
                cf_params[key] = kwargs[key]
            elif key in model.params:
                cf_params[key] = model.params[key]
            elif key in cf_params:
                continue  # Skip if already provided (e.g., spot, t, r, q)
            else:
                raise ValueError(
                    f"Required CF parameter '{key}' not found in kwargs or model.params."
                )
        phi = model.cf(**cf_params)

        # Form u = v - 0.5i and integrand
        v = np.arange(self.N) * eta
        u = v - 0.5j
        denom = u * u + 0.25
        with np.errstate(divide="ignore", invalid="ignore"):
            integrand = np.exp(-r * T) * phi(u) / denom
        integrand[0] = 0.0  # Handle singularity at v = 0

        w = np.ones(self.N)
        w[1:-1:2] = 4
        w[2:-2:2] = 2
        weights = w * eta / 3.0

        fft_input = integrand * np.exp(1j * v * (k_grid[0] - np.log(K))) * weights
        fft_vals = np.fft.fft(fft_input).real

        call_price_grid = np.exp(-0.5 * k_grid) * fft_vals / math.pi

        # Interpolate to target strike
        call_price = float(np.interp(np.log(K), k_grid, call_price_grid))

        price = (
            call_price
            if is_call
            else call_price - S * np.exp(-q * T) + K * np.exp(-r * T)
        )

        greeks = {}
        try:
            phi_minus_half_i = phi(-0.5j)
            if abs(phi_minus_half_i) > 1e-12:
                delta_integrand = (
                    np.exp(-r * T) * phi(u - 1j) / (phi_minus_half_i * denom)
                )
                delta_fft_input = (
                    delta_integrand * np.exp(1j * v * (k_grid[0] - np.log(K))) * weights
                )
                delta_fft_vals = np.fft.fft(delta_fft_input).real
                delta_grid = np.exp(-0.5 * k_grid) * delta_fft_vals / math.pi
                delta = float(np.interp(np.log(K), k_grid, delta_grid))
                delta = 0.5 + delta if is_call else 0.5 + delta - np.exp(-q * T)
                greeks["delta"] = delta
        except (ValueError, RuntimeError):
            pass  # Fallback to GreekMixin

        return PricingResult(price=float(price), Greeks=greeks)

    @staticmethod
    def _vol_proxy(model: BaseModel, kw: dict[str, Any]) -> float:
        """
        Best-effort volatility proxy used for eta heuristic.

        Parameters
        ----------
        model : BaseModel
            Pricing model.
        kw : dict[str, Any]
            Additional model parameters.

        Returns
        -------
        float
            Volatility proxy.
        """
        if model.name.lower() == "variancegamma":
            return 0.2  # Use model sigma for VG
        if "sigma" in model.params and model.params["sigma"] is not None:
            return model.params["sigma"]
        if "v0" in kw and kw["v0"] is not None:
            return math.sqrt(kw["v0"])
        if "v0" in model.params and model.params["v0"] is not None:
            return math.sqrt(kw["v0"])
        if "vol_of_vol" in model.params and model.params["vol_of_vol"] is not None:
            return model.params["vol_of_vol"]
        return 0.3
