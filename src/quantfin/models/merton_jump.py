from __future__ import annotations
import math
import numpy as np
from scipy.stats import norm
from scipy.special import factorial
from typing import Any

from quantfin.models.base import BaseModel, ParamValidator, CF


class MertonJumpModel(BaseModel):
    """
    Merton Jump-Diffusion model for option pricing, implemented as an infinite sum
    of BSM prices weighted by Poisson probabilities.
    """
    name = "Merton Jump-Diffusion"
    supports_cf = True
    has_closed_form = True
    supports_sde = True
    has_jumps = True
    cf_kwargs = BaseModel.cf_kwargs

    def _validate_params(self) -> None:
        p = self.params
        ParamValidator.require(p, ["lambda", "mu_j", "sigma_j", "sigma", "max_sum_terms"], model=self.name)
        ParamValidator.positive(p, ["lambda", "sigma_j", "sigma", "max_sum_terms"], model=self.name)

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
        Merton's closed-form solution as a sum of BSM prices.

        Parameters
        ----------
        spot : float
            Underlying spot price.
        strike : float
            Strike price.
        r : float
            Risk-free rate.
        q : float
            Dividend yield.
        t : float
            Time to maturity in years.
        call : bool
            True for call, False for put.

        Returns
        -------
        float
            Option price.
        """
        if t <= 0 or spot <= 0 or self.params["sigma"] <= 0 or self.params["sigma_j"] <= 0:
            return max(0.0, spot - strike) if call else max(0.0, strike - spot)

        lambda_ = self.params["lambda"]
        mu_j = self.params["mu_j"]
        sigma_j = self.params["sigma_j"]
        sigma = self.params["sigma"]
        max_sum_terms = self.params["max_sum_terms"]

        k = math.exp(mu_j + 0.5 * sigma_j**2) - 1
        lambda_prime = lambda_ * (1 + k)
        yMul = lambda_prime * t
        factor = np.exp(-yMul)
        interval = np.arange(max_sum_terms)

        # Vectorized computation
        r_n = r - lambda_ * k + (interval * (mu_j + 0.5 * sigma_j**2)) / t
        sigma_n = np.sqrt(sigma**2 + (interval * sigma_j**2) / t)
        weights = (yMul ** interval) / factorial(interval)

        # Early termination: mask terms with negligible weight
        mask = (weights > 1e-12)
        mean_jumps = lambda_prime * t
        hard_cap = int(mean_jumps + 10 * math.sqrt(mean_jumps))
        mask &= (interval <= hard_cap)
        
        weights = weights[mask]
        r_n = r_n[mask]
        sigma_n = sigma_n[mask]

        bsm_values = self._bsm_price(spot, strike, r_n, t, sigma_n, q, call)
        return factor * np.sum(weights * bsm_values)

    def _bsm_price(
        self,
        spot: float,
        strike: float,
        r: np.ndarray | float,
        t: float,
        sigma: np.ndarray | float,
        q: float,
        call: bool = True,
    ) -> np.ndarray | float:
        """Vectorized Black-Scholes-Merton price."""

        sqrt_t = np.sqrt(t)
        sigma = np.maximum(sigma, 1e-12) 
        d1 = (np.log(spot / strike) + (r - q + 0.5 * sigma**2) * t) / (sigma * sqrt_t)
        d2 = d1 - sigma * sqrt_t
        df_div = np.exp(-q * t)
        df_rate = np.exp(-r * t)
        if call:
            return spot * df_div * norm.cdf(d1) - strike * df_rate * norm.cdf(d2)
        else:
            return strike * df_rate * norm.cdf(-d2) - spot * df_div * norm.cdf(-d1)

    def _cf_impl(self, *, t: float, spot: float, r: float, q: float) -> CF:
        """
        Merton Jump-Diffusion characteristic function for log-spot price log(S_t).
        """
        p = self.params
        sigma, lambda_, mu_j, sigma_j = p["sigma"], p["lambda"], p["mu_j"], p["sigma_j"]

        # Drift of the BSM part
        drift = r - q - 0.5 * sigma**2
        # Jump compensator (to ensure risk-neutrality)
        compensator = lambda_ * (np.exp(mu_j + 0.5 * sigma_j**2) - 1)

        def phi(u: np.ndarray | complex) -> np.ndarray | complex:
            # BSM component
            bsm_part = 1j * u * (np.log(spot) + (drift - compensator) * t) - 0.5 * u**2 * sigma**2 * t
            # Jump component
            jump_part = lambda_ * t * (np.exp(1j * u * mu_j - 0.5 * u**2 * sigma_j**2) - 1)
            
            return np.exp(bsm_part + jump_part)

        return phi

    def _sde_impl(self):
        p = self.params
        sigma, mu_j, sigma_j = p["sigma"], p["mu_j"], p["sigma_j"]
        lambda_ = p["lambda"]
        
        # Pre-calculate the risk-neutral compensator
        compensator = lambda_ * (np.exp(mu_j + 0.5 * sigma_j**2) - 1)

        def stepper(
            log_s_t: np.ndarray,
            r: float,
            q: float,
            dt: float,
            dw_t: np.ndarray,
            jump_counts: np.ndarray,
            rng: np.random.Generator
        ) -> np.ndarray:
            
            # Calculate the full drift for the continuous part
            drift_term = (r - q - 0.5 * sigma**2 - compensator) * dt

            # 1. Evolve with the continuous diffusion part first
            next_log_s = log_s_t + drift_term + sigma * dw_t
            
            # 2. Add the jumps for paths where they occurred
            paths_with_jumps = np.where(jump_counts > 0)[0]
            
            for path_idx in paths_with_jumps:
                num_jumps_on_path = jump_counts[path_idx]
                
                # Draw the jump sizes from a normal distribution and add to the path
                total_jump_size = np.sum(rng.normal(loc=mu_j, scale=sigma_j, size=num_jumps_on_path))
                
                next_log_s[path_idx] += total_jump_size

            return next_log_s

        return stepper

    def _pde_impl(self) -> Any:
        raise NotImplementedError(f"{self.name} PDE not implemented")