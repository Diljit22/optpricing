from __future__ import annotations
import numpy as np
from scipy.stats import norm
from typing import Any, Callable, Tuple

from quantfin.models.base import BaseModel, ParamValidator, CF, PDECoeffs

class BSMModel(BaseModel):
    """
    Black-Scholes-Merton model:
      - supports characteristic function (CF)
      - supports SDE sampling
      - supports PDE solvers
      - provides closed-form price and analytic delta
      - supports standard lattices: CRR, Leisen-Reimer, TOPM
    """
    name = "Black-Scholes-Merton"
    supports_cf = True
    supports_sde = True
    supports_pde = True
    has_closed_form = True
    supported_lattices = {"crr", "lr", "topm"}
    cf_kwargs = BaseModel.cf_kwargs

    def _validate_params(self) -> None:
        p = self.params
        ParamValidator.require(p, ["sigma"], model=self.name)
        ParamValidator.positive(p, ["sigma"], model=self.name)

    def _closed_form_impl(
        self,
        *,
        spot: float,
        strike: float,
        r: float,
        q: float,
        t: float,
        call: bool = True,
    ) -> float:
        """
        Compute the Black-Scholes-Merton price in closed form.
        """
        sigma = self.params["sigma"]
        sqrt_t = np.sqrt(t)
        d1 = (np.log(spot / strike) + (r - q + 0.5 * sigma * sigma) * t) / (sigma * sqrt_t)
        d2 = d1 - sigma * sqrt_t

        df_div = np.exp(-q * t)
        df_rate = np.exp(-r * t)

        if call:
            return spot * df_div * norm.cdf(d1) - strike * df_rate * norm.cdf(d2)
        else:
            return strike * df_rate * norm.cdf(-d2) - spot * df_div * norm.cdf(-d1)

    def delta_analytic(
        self,
        *,
        spot: float,
        strike: float,
        r: float,
        q: float,
        t: float,
        call: bool = True,
    ) -> float:
        """
        Analytic delta for BSM: derivative of price with respect to spot.
        """
        sigma = self.params["sigma"]
        sqrt_t = np.sqrt(t)
        d1 = (np.log(spot / strike) + (r - q + 0.5 * sigma * sigma) * t) / (sigma * sqrt_t)
        df_div = np.exp(-q * t)
        return df_div * norm.cdf(d1) if call else -df_div * norm.cdf(-d1)

    def gamma_analytic(
        self,
        *,
        spot:   float,
        strike: float,
        r:      float,
        q:      float,
        t:      float,
    ) -> float:
        sigma = self.params["sigma"]
        sqrt_t = np.sqrt(t)
        d1 = (np.log(spot / strike) + (r - q + 0.5*sigma**2)*t) / (sigma*sqrt_t)
        df_div = np.exp(-q * t)
        return df_div * norm.pdf(d1) / (spot * sigma * sqrt_t)

    def vega_analytic(
        self,
        *,
        spot:   float,
        strike: float,
        r:      float,
        q:      float,
        t:      float,
    ) -> float:
        sigma = self.params["sigma"]
        sqrt_t = np.sqrt(t)
        d1 = (np.log(spot / strike) + (r - q + 0.5*sigma**2)*t) / (sigma*sqrt_t)
        return spot * np.exp(-q*t) * norm.pdf(d1) * sqrt_t

    def theta_analytic(
        self,
        *,
        spot:   float,
        strike: float,
        r:      float,
        q:      float,
        t:      float,
        call:   bool = True,
    ) -> float:
        sigma = self.params["sigma"]
        sqrt_t = np.sqrt(t)
        d1 = (np.log(spot / strike) + (r - q + 0.5*sigma**2)*t) / (sigma*sqrt_t)
        d2 = d1 - sigma*sqrt_t
        df_div  = np.exp(-q * t)
        df_rate = np.exp(-r * t)
        pdf_d1  = norm.pdf(d1)

        term1 = -spot * df_div * pdf_d1 * sigma / (2*sqrt_t)
        if call:
            term2 =  q * spot * df_div * norm.cdf(d1)
            term3 = -r * strike * df_rate * norm.cdf(d2)
        else:
            term2 = -q * spot * df_div * norm.cdf(-d1)
            term3 =  r * strike * df_rate * norm.cdf(-d2)
        return term1 + term2 + term3

    def rho_analytic(
        self,
        *,
        spot:   float,
        strike: float,
        r:      float,
        q:      float,
        t:      float,
        call:   bool = True,
    ) -> float:
        df_rate = np.exp(-r*t)
        d2 = (np.log(spot/strike) + (r - q - 0.5*self.params["sigma"]**2)*t)/ (self.params["sigma"]*np.sqrt(t))
        return strike * t * df_rate * (norm.cdf(d2) if call else -norm.cdf(-d2))

    def _cf_impl(self, *, t: float, spot: float, r: float, q: float) -> CF:
        """
        Returns the characteristic function phi(u) for the log-spot price log(S_t).
        This is defined as E[exp(i * u * log(S_t))].
        """
        sigma = self.params["sigma"]
        
        # The drift of the log-price process in the risk-neutral world
        drift = r - q - 0.5 * sigma**2
        
        # The characteristic function of the log-price process X_t = log(S_t)
        def phi(u: np.ndarray | complex) -> np.ndarray | complex:
            # The process is log(S_t) = log(S_0) + (r - q - 0.5*sigma^2)*t + sigma*W_t
            # E[exp(i*u*log(S_t))] = exp(i*u*log(S_0)) * E[exp(i*u*[(r-q-0.5s^2)t + sW_t])]
            # The expectation of the geometric Brownian motion part is known.
            mean_component = 1j * u * (np.log(spot) + drift * t)
            variance_component = -0.5 * (u**2) * (sigma**2) * t
            exponent = mean_component + variance_component
            if isinstance(exponent, np.ndarray): exponent.real = np.clip(exponent.real, -700, 700)
            elif np.isreal(exponent): exponent = np.clip(exponent, -700, 700)
            return np.exp(exponent)
            
        return phi

    def _sde_impl(self) -> Callable:
        """
        Returns the Euler-Maruyama stepper for the BSM log-price process.
        This is a single-factor SDE.

        Signature: (log_s_t, r, q, dt, dw_t) -> next_log_s_t
        """
        sigma = self.params["sigma"]

        def stepper(
            log_s_t: np.ndarray, # Current log-spot price
            r: float,
            q: float,
            dt: float,
            dw_t: np.ndarray # Random draws ~ N(0, sqrt(dt))
        ) -> np.ndarray:
            drift = (r - q - 0.5 * sigma**2) * dt
            diffusion = sigma * dw_t
            return log_s_t + drift + diffusion

        return stepper

    def _pde_impl(self) -> PDECoeffs:
        """
        Returns the Black-Scholes PDE coefficients A(S), B(S), C(S) for use in
        a generic PDE solver.
        """
        sigma = self.params["sigma"]
        def coeffs(
            S: np.ndarray, r: float, q: float
            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            A = 0.5 * sigma**2 * S**2
            B = (r - q) * S
            C = -r * np.ones_like(S)
            return A, B, C
        return coeffs
    