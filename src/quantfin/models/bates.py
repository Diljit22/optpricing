from __future__ import annotations
import numpy as np
from typing import Any, Callable

from quantfin.models.base import BaseModel, ParamValidator, CF

class BatesModel(BaseModel):
    """
    Bates (1996) stochastic volatility jump-diffusion model (Heston + Merton Jumps).
    """
    name = "Bates"
    supports_cf = True
    supports_sde = True
    has_jumps = True
    has_variance_process = True
    cf_kwargs = BaseModel.cf_kwargs + ("v0",)

    def _validate_params(self) -> None:
        p = self.params
        req = ["kappa", "theta", "rho", "vol_of_vol", "lambda", "mu_j", "sigma_j"]
        ParamValidator.require(p, req, model=self.name)
        # Add other validations as in Heston and Merton...

    def _cf_impl(self, *, t: float, spot: float, r: float, q: float, v0: float) -> CF:
        """
        Bates characteristic function for log-spot price log(S_t).
        """
        p = self.params
        kappa, theta, rho, vol_of_vol = p["kappa"], p["theta"], p["rho"], p["vol_of_vol"]
        lambda_, mu_j, sigma_j = p["lambda"], p["mu_j"], p["sigma_j"]

        # Jump compensator
        compensator = lambda_ * (np.exp(mu_j + 0.5 * sigma_j**2) - 1)

        def phi(u: np.ndarray | complex) -> np.ndarray | complex:
            # Heston part (with adjusted drift for jump compensator)
            d = np.sqrt((rho * vol_of_vol * u * 1j - kappa)**2 - (vol_of_vol**2) * (-u * 1j - u**2))
            g = (kappa - rho * vol_of_vol * u * 1j - d) / (kappa - rho * vol_of_vol * u * 1j + d)
            C = (r - q - compensator) * u * 1j * t + (kappa * theta / vol_of_vol**2) * (
                (kappa - rho * vol_of_vol * u * 1j - d) * t - 2 * np.log((1 - g * np.exp(-d * t)) / (1 - g))
            )
            D = ((kappa - rho * vol_of_vol * u * 1j - d) / vol_of_vol**2) * ((1 - np.exp(-d * t)) / (1 - g * np.exp(-d * t)))
            heston_part = np.exp(C + D * v0 + 1j * u * np.log(spot))

            # Merton jump part
            jump_part = np.exp(lambda_ * t * (np.exp(1j * u * mu_j - 0.5 * u**2 * sigma_j**2) - 1))

            return heston_part * jump_part

        return phi
    
    def _sde_impl(self) -> Callable:
        p = self.params
        kappa, theta, rho, vol_of_vol = p["kappa"], p["theta"], p["rho"], p["vol_of_vol"]
        lambda_, mu_j, sigma_j = p["lambda"], p["mu_j"], p["sigma_j"]
        rho_bar = np.sqrt(1 - rho**2)
        
        compensator = lambda_ * (np.exp(mu_j + 0.5 * sigma_j**2) - 1)

        def stepper(
            log_s_t: np.ndarray,
            v_t: np.ndarray,
            r: float,
            q: float,
            dt: float,
            dw_s: np.ndarray,
            dw_v: np.ndarray,
            jump_counts: np.ndarray,
            rng: np.random.Generator
        ) -> tuple[np.ndarray, np.ndarray]:
            
            v_t_pos = np.maximum(v_t, 0)
            
            # Evolve variance (Heston part)
            v_t_next = np.abs(v_t + kappa * (theta - v_t_pos) * dt + vol_of_vol * np.sqrt(v_t_pos) * dw_v)
            
            # Evolve log-spot (Heston part + jump compensator)
            s_drift = (r - q - 0.5 * v_t_pos - compensator) * dt
            s_diffusion = np.sqrt(v_t_pos) * (rho * dw_v + rho_bar * dw_s)
            next_log_s = log_s_t + s_drift + s_diffusion
            
            # Add jumps (Merton part)
            paths_with_jumps = np.where(jump_counts > 0)[0]
            for path_idx in paths_with_jumps:
                num_jumps = jump_counts[path_idx]
                jump_size = np.sum(rng.normal(loc=mu_j, scale=sigma_j, size=num_jumps))
                next_log_s[path_idx] += jump_size
            
            return next_log_s, v_t_next

        return stepper

    def _pde_impl(self) -> Any:
        raise NotImplementedError(f"{self.name} PDE not implemented")
    
    def _closed_form_impl(self) -> Any:
        raise NotImplementedError(f"{self.name} Closed Form not implemented")
