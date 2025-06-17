from __future__ import annotations
import numpy as np
from scipy import integrate
from typing import Any

from techniques.base.base_technique import BaseTechnique, PricingResult
from techniques.base.greek_mixin import GreekMixin
from techniques.base.iv_mixin import IVMixin
from models.base.base_model import BaseModel
from atoms.option import Option, OptionType
from atoms.stock import Stock
from atoms.rate import Rate

class IntegrationTechnique(BaseTechnique, GreekMixin, IVMixin):
    """
    Prices options using the Gil-Pelaez inversion formula, calculated via
    numerical quadrature. This technique is vectorized for performance.

    It calculates the price and delta "for free" simultaneously.
    """
    def __init__(self, *, upper_bound_mult: float = 20.0, limit: int = 200, epsabs: float = 1e-9, epsrel: float = 1e-9):
        """
        upper_bound_mult: Multiplier for dynamic upper bound calculation.
                          Bound will be ~ mult / (sigma * sqrt(T)).
        """
        self.upper_bound_mult = upper_bound_mult
        self.limit = limit
        self.epsabs = epsabs
        self.epsrel = epsrel

    def price(self, option: Option, stock: Stock, model: BaseModel, rate: Rate, **kwargs: Any) -> PricingResult:
        """
        Calculates the option price and delta using vectorized numerical integration.
        """
        if not model.supports_cf:
            raise TypeError(f"Model '{model.name}' does not support a characteristic function.")

        S, K, T = stock.spot, option.strike, option.maturity
        r, q = rate.rate, stock.dividend

        vol_proxy = model.params.get("sigma")
        if vol_proxy is None:
            vol_proxy = kwargs.get("v0")          # Heston v0
            if vol_proxy is not None:
                vol_proxy = np.sqrt(vol_proxy)
        if vol_proxy is None:
            vol_proxy = model.params.get("vol_of_vol")
        if vol_proxy is None:
            vol_proxy = 0.30                      # final fallback

        # 2. Choose the cut-off
        if getattr(model, "is_pure_levy", False):
            # Heavy-tailed CF → integrate much farther out
            upper_bound = 250.0 / max(T, 1e-8)    # e.g. 250 for T = 1
        else:
            # Gaussian-tail heuristic (previous logic)
            if vol_proxy * T <= 0:
                upper_bound = 200.0
            else:
                upper_bound = self.upper_bound_mult / (vol_proxy * np.sqrt(T))
        
        # Get the characteristic function for the log-price
        cf_params = {"t": T, "spot": S, "r": r, "q": q}
        for key in getattr(model, "cf_kwargs", ()):
            if key in model.params and key not in cf_params:
                cf_params[key] = model.params[key]
        cf_params.update(kwargs)
        phi = model.cf(**cf_params)

        # --- Vectorized Integration using quad_vec ---
        # We integrate a single strike K, so we wrap it in an array.
        # The logic is built to handle arrays of strikes if needed later.
        k_log = np.log(np.array([K]))

        # Integrand for P2 (risk-neutral probability of S_T > K)
        # Note: phi(u) is the CF of log(S_T). We need CF of log(S_T/K).
        # exp(-1j*u*logK) * phi(u) is the CF of log(S_T) - log(K) = log(S_T/K)
        integrand_p2 = lambda u: (np.imag(np.exp(-1j * u * k_log[:, None]) * phi(u)) / u).flatten()

        # Integrand for P1 (delta-related probability)
        # This requires the "twisted" characteristic function phi(u-i)
        phi_minus_i = phi(-1j)
        if abs(phi_minus_i) < 1e-12: # Avoid division by zero
            integrand_p1 = lambda u: np.zeros_like(u)
        else:
            integrand_p1 = lambda u: (np.imag(np.exp(-1j * u * k_log[:, None]) * phi(u - 1j) / phi_minus_i) / u).flatten()

        # Perform the integrations
        # Note: quad_vec returns (integral_result, error_estimate)
        integral_p2, _ = integrate.quad_vec(integrand_p2, 1e-15, upper_bound, limit=self.limit, epsabs=self.epsabs, epsrel=self.epsrel)
        integral_p1, _ = integrate.quad_vec(integrand_p1, 1e-15, upper_bound, limit=self.limit, epsabs=self.epsabs, epsrel=self.epsrel)

        # Gil-Pelaez Inversion Formulas
        P2 = 0.5 + integral_p2[0] / np.pi
        P1 = 0.5 + integral_p1[0] / np.pi

        # Final pricing formula (same as BSM, but with general probabilities)
        df_S = S * np.exp(-q * T)
        df_K = K * np.exp(-r * T)

        if option.option_type is OptionType.CALL:
            price = df_S * P1 - df_K * P2
            delta = np.exp(-q * T) * P1
        else: # Put
            price = df_K * (1 - P2) - df_S * (1 - P1)
            delta = np.exp(-q * T) * (P1 - 1.0)
            
        return PricingResult(price=price, Greeks={'delta': delta})

    def delta(self, option: Option, stock: Stock, model: BaseModel, rate: Rate, **kwargs: Any) -> float:
        """
        Overrides the GreekMixin's finite difference method.
        Returns the 'free' delta calculated during the pricing call.
        """
        # To get the free delta, we must call the price method.
        result = self.price(option, stock, model, rate, **kwargs)
        return result.Greeks['delta']