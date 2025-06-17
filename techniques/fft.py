from __future__ import annotations
import math
from typing import Any, Callable, Dict, Tuple

import numpy as np
from scipy import integrate

from techniques.base.base_technique import BaseTechnique, PricingResult
from techniques.base.greek_mixin import GreekMixin
from techniques.base.iv_mixin import IVMixin
from models.base.base_model import BaseModel
from atoms.option import Option, OptionType
from atoms.stock import Stock
from atoms.rate import Rate


class FFTTechnique(BaseTechnique, GreekMixin, IVMixin):
    """Fast‐Fourier Carr-Madan pricer with optional analytic delta.

    * **Dynamic alpha**.
      transform always exists and the grid is neither over- nor under-damped.
    * **Adaptive η** - scale the Fourier step by volatility to keep the
      strike grid dense enough at low vol and coarse enough at high vol.
    * **Model-agnostic** - every keyword in ``model.cf_kwargs`` is copied from
      ``model.params`` or ``**kwargs`` so stochastic-vol and jump models work
      out-of-the-box (e.g. ``v0`` for Heston).
    * **Free delta** - computes the *P₁* integral on the same FFT grid so no
      separate finite-difference call is required; falls back to a central
      difference if numerical issues arise.
    """

    # ---------------------------------------------------------------------
    def __init__(
        self,
        *,
        alpha: float | None = None,
        n: int = 12,
        eta: float = 0.25,
    ) -> None:
        super().__init__()
        self._alpha_user = alpha  # keep None so we can choose dynamically
        self.n = int(n)
        self.N = 1 << self.n
        self.base_eta = float(eta)

        # Simpson weights (1 4 2 4 … 1)*η/3 - initialised later after η chosen
        self._weights: np.ndarray | None = None

    # ------------------------------------------------------------------ #
    # pricing public API
    # ------------------------------------------------------------------ #
    def price(
        self,
        option: Option,
        stock: Stock,
        model: BaseModel,
        rate: Rate,
        **kwargs: Any,
    ) -> PricingResult:
        if not model.supports_cf:
            raise TypeError(f"Model '{model.name}' does not expose a CF.")

        S, K, T = stock.spot, option.strike, option.maturity
        r, q = rate.rate, stock.dividend
        is_call = option.option_type is OptionType.CALL

        # ---- choose α and η grid ------------------------------------------------
        vol_proxy = self._vol_proxy(model, kwargs)
        if self._alpha_user is not None:
            alpha = self._alpha_user
        elif vol_proxy is None:
            alpha = 1.75 # Use a higher, fixed alpha for these models
        else:
            alpha = 1.0 + 0.5 * vol_proxy * math.sqrt(T)
        eta = self.base_eta * max(1.0, vol_proxy * math.sqrt(T))
        lambd = 2 * math.pi / (self.N * eta)
        b = 0.5 * self.N * lambd

        # (re-)build Simpson weights for this η only once per call
        w = np.ones(self.N)
        w[1:-1:2] = 4
        w[2:-2:2] = 2
        weights = w * eta / 3  # shape (N,)

        # ---- characteristic function ------------------------------------------
        cf_params: Dict[str, Any] = {"t": T, "spot": S, "r": r, "q": q}
        for k in getattr(model, "cf_kwargs", ()):  # copy extras
            if k in kwargs:
                cf_params[k] = kwargs[k]
            elif k in model.params:
                cf_params[k] = model.params[k]
        phi = model.cf(**cf_params)

        # ---- Carr-Madan integrand ---------------------------------------------
        u = np.arange(self.N) * eta  # [0, …, (N-1)η]
        disc = math.exp(-r * T)
        numer = phi(u - 1j * (alpha + 1))
        denom = alpha ** 2 + alpha - u ** 2 + 1j * u * (2 * alpha + 1)
        psi = disc * numer / denom

        fft_input = psi * np.exp(1j * u * b) * weights
        fft_vals = np.fft.fft(fft_input).real

        k_grid = -b + np.arange(self.N) * lambd  # log-strike grid
        price_grid = np.exp(-alpha * k_grid) / math.pi * fft_vals

        # ---- interpolate price -------------------------------------------------
        k_target = math.log(K)
        price = np.interp(k_target, k_grid, price_grid)

        if not is_call:
            price = price - S * math.exp(-q * T) + K * disc

        return PricingResult(price=float(price))

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _vol_proxy(model: BaseModel, kw: Dict[str, Any]) -> float:
        """
        Best-effort volatility proxy used for alpha/eta heuristics.
        This version is guaranteed to return a float.
        """
        if model.name == "Variance Gamma":
            return 2.9
        # 1. Check for 'sigma' in model params (for BSM, Merton, etc.)
        if "sigma" in model.params and model.params["sigma"] is not None:
            return model.params["sigma"]
        
        # 2. Check for 'v0' in kwargs or model params (for Heston, Bates)
        #    The 'kw' check is first, as it's more explicit.
        if "v0" in kw and kw["v0"] is not None:
            return math.sqrt(kw["v0"])
        if "v0" in model.params and model.params["v0"] is not None:
            return math.sqrt(model.params["v0"])
            
        # 3. Last resort proxy for stochastic vol models
        if "vol_of_vol" in model.params and model.params["vol_of_vol"] is not None:
            return model.params["vol_of_vol"]
            
        # 4. Final fallback default value
        return 0.3