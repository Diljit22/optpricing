
from __future__ import annotations
import math
import numpy as np
from typing import Any, Dict

from quantfin.atoms import Option, OptionType, Stock, Rate
from quantfin.models import BaseModel
from quantfin.techniques.base import BaseTechnique, PricingResult, GreekMixin, IVMixin

class FFTTechnique(BaseTechnique, GreekMixin, IVMixin):
    """
    Fast Fourier Transform (FFT) pricer based on the Carr-Madan formula,
    preserving the original tuned logic for grid and parameter selection.
    """
    def __init__(self, *, n: int = 12, eta: float = 0.25, alpha: float | None = None):
        self.n = int(n)
        self.N = 1 << self.n
        self.base_eta = float(eta)
        self.alpha_user = alpha
        self._cached_results: dict[str, Any] = {}

    def _price_and_greeks(self, option: Option, stock: Stock, model: BaseModel, rate: Rate, **kwargs: Any) -> Dict[str, float]:
        """Internal method to perform the core FFT calculation once."""
        if not model.supports_cf:
            raise TypeError(f"Model '{model.name}' does not support a characteristic function.")

        S0, K, T = stock.spot, option.strike, option.maturity
        r, q = rate.get_rate(T), stock.dividend

        # 1. Set up FFT grid parameters using YOUR original tuned logic
        vol_proxy = self._get_vol_proxy(model, kwargs)
        
        if self.alpha_user is not None:
            alpha = self.alpha_user
        elif vol_proxy is None:
            alpha = 1.75
        else:
            alpha = 1.0 + 0.5 * vol_proxy * math.sqrt(T)
            
        eta = self.base_eta * max(1.0, vol_proxy * math.sqrt(T)) if vol_proxy is not None else self.base_eta
        
        lambda_ = (2 * math.pi) / (self.N * eta)
        b = (self.N * lambda_) / 2.0
        k_grid = -b + lambda_ * np.arange(self.N)

        # 2. Set up Simpson's rule weights
        w = np.ones(self.N)
        w[1:-1:2], w[2:-2:2] = 4, 2
        weights = w * eta / 3.0

        # 3. Get the model's characteristic function
        phi = model.cf(t=T, spot=S0, r=r, q=q, **kwargs)

        # 4. Calculate FFT for the call price using YOUR original formula
        u = np.arange(self.N) * eta
        discount = math.exp(-r * T)
        numerator = phi(u - 1j * (alpha + 1))
        denominator = alpha**2 + alpha - u**2 + 1j * u * (2 * alpha + 1)
        psi = discount * numerator / denominator

        fft_input = psi * np.exp(1j * u * b) * weights
        fft_vals = np.fft.fft(fft_input).real
        
        # Note: The original formula priced C*exp(alpha*k). We need to divide by exp(alpha*k).
        call_price_grid = np.exp(-alpha * k_grid) / math.pi * fft_vals

        # 5. Interpolate to find the results at the target strike
        k_target = math.log(K)
        call_price = np.interp(k_target, k_grid, call_price_grid)

        # 6. Use put-call parity for put price
        if option.option_type is OptionType.CALL:
            price = call_price
        else: # Put
            price = call_price - (S0 * np.exp(-q * T) - K * np.exp(-r * T))
            
        return {"price": price}

    def price(self, option: Option, stock: Stock, model: BaseModel, rate: Rate, **kwargs: Any) -> PricingResult:
        """Calculates the option price using the FFT method."""
        self._cached_results = self._price_and_greeks(option, stock, model, rate, **kwargs)
        return PricingResult(price=self._cached_results['price'])

    # Note: For FFT, we let the GreekMixin handle all Greeks via finite difference.
    # This is a robust choice as analytic FFT greeks can be complex to implement.
    # The caching in the mixin is not used here, as each Greek call is a full price recalculation.

    @staticmethod
    def _get_vol_proxy(model: BaseModel, kw: Dict[str, Any]) -> float | None:
        """Best-effort volatility proxy used for grid heuristics."""
        if "sigma" in model.params and model.params["sigma"] is not None:
            return model.params["sigma"]
        if "v0" in kw and kw["v0"] is not None:
            return math.sqrt(kw["v0"])
        if "v0" in model.params and model.params["v0"] is not None:
            return math.sqrt(model.params["v0"])
        if "vol_of_vol" in model.params and model.params["vol_of_vol"] is not None:
            return model.params["vol_of_vol"]
        return None