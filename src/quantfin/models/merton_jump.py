from __future__ import annotations
import math
import numpy as np
from scipy.special import factorial
from typing import Any

from quantfin.models.base import BaseModel, ParamValidator, CF
from quantfin.models.bsm import BSMModel

class MertonJumpModel(BaseModel):
    """
    Merton Jump-Diffusion model, implemented as a Poisson-weighted sum of BSM prices.
    """
    name: str = "Merton Jump-Diffusion"
    supports_cf: bool = True
    has_closed_form: bool = True
    supports_sde: bool = True
    has_jumps: bool = True

    def __init__(self, params: dict[str, float]):
        super().__init__(params)
        # Composition: Create a BSMModel instance for the core diffusion part.
        self.bsm_solver = BSMModel(params={"sigma": self.params["sigma"]})

    def _validate_params(self) -> None:
        p = self.params
        req = ["lambda", "mu_j", "sigma_j", "sigma", "max_sum_terms"]
        ParamValidator.require(p, req, model=self.name)
        ParamValidator.positive(p, ["lambda", "sigma_j", "sigma", "max_sum_terms"], model=self.name)

    def _closed_form_impl(self, *, spot: float, strike: float, r: float, q: float, t: float, call: bool = True) -> float:
        """Merton's closed-form solution as a sum of BSM prices."""
        lambda_ = self.params["lambda"]
        mu_j = self.params["mu_j"]
        sigma_j = self.params["sigma_j"]
        max_sum_terms = int(self.params["max_sum_terms"])

        k = math.exp(mu_j + 0.5 * sigma_j**2) - 1
        lambda_prime = lambda_ * (1 + k)
        y_mul = lambda_prime * t
        
        interval = np.arange(max_sum_terms)
        weights = np.exp(-y_mul) * (y_mul ** interval) / factorial(interval)

        # Vectorized computation of adjusted BSM parameters
        r_n = r - lambda_ * k + (interval * (mu_j + 0.5 * sigma_j**2)) / t
        sigma_n_sq = self.params["sigma"]**2 + (interval * sigma_j**2) / t
        sigma_n = np.sqrt(np.maximum(sigma_n_sq, 1e-12))

        total_price = 0.0
        for i in range(max_sum_terms):
            # Terminate sum early if weights become negligible
            if weights[i] < 1e-12 and i > lambda_prime * t:
                break
            
            # Create a temporary BSM model with the adjusted volatility
            temp_bsm_solver = self.bsm_solver.with_params(sigma=sigma_n[i])
            
            price_term = temp_bsm_solver.price_closed_form(spot=spot, strike=strike, r=r_n[i], q=q, t=t, call=call)
            total_price += weights[i] * price_term
            
        return total_price

    def _cf_impl(self, *, t: float, spot: float, r: float, q: float) -> CF:
        p = self.params
        sigma, lambda_, mu_j, sigma_j = p["sigma"], p["lambda"], p["mu_j"], p["sigma_j"]
        drift = r - q - 0.5 * sigma**2
        compensator = lambda_ * (np.exp(mu_j + 0.5 * sigma_j**2) - 1)
        def phi(u: np.ndarray | complex) -> np.ndarray | complex:
            bsm_part = 1j * u * (np.log(spot) + (drift - compensator) * t) - 0.5 * u**2 * sigma**2 * t
            jump_part = lambda_ * t * (np.exp(1j * u * mu_j - 0.5 * u**2 * sigma_j**2) - 1)
            return np.exp(bsm_part + jump_part)
        return phi

    def _sde_impl(self):
        p = self.params
        sigma, mu_j, sigma_j, lambda_ = p["sigma"], p["mu_j"], p["sigma_j"], p["lambda"]
        compensator = lambda_ * (np.exp(mu_j + 0.5 * sigma_j**2) - 1)
        def stepper(log_s_t: np.ndarray, r: float, q: float, dt: float, dw_t: np.ndarray, jump_counts: np.ndarray, rng: np.random.Generator) -> np.ndarray:
            drift_term = (r - q - 0.5 * sigma**2 - compensator) * dt
            next_log_s = log_s_t + drift_term + sigma * dw_t
            paths_with_jumps = np.where(jump_counts > 0)[0]
            for path_idx in paths_with_jumps:
                num_jumps_on_path = jump_counts[path_idx]
                total_jump_size = np.sum(rng.normal(loc=mu_j, scale=sigma_j, size=num_jumps_on_path))
                next_log_s[path_idx] += total_jump_size
            return next_log_s
        return stepper

    def _pde_impl(self) -> Any:
        raise NotImplementedError(f"{self.name} PDE not implemented")