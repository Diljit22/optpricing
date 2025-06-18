from __future__ import annotations
from typing import Any, Callable
import numpy as np

from quantfin.models.base import BaseModel, ParamValidator, CF


class SABRJumpModel(BaseModel):
    """
    SABR model with an added jump component on the spot process.
    """
    name = "SABR with Jumps"
    supports_sde = True
    has_variance_process = True
    has_jumps = True

    def _validate_params(self) -> None:
        p = self.params
        # SABR params + Jump params
        req = ["alpha", "beta", "rho", "lambda", "mu_j", "sigma_j"]
        ParamValidator.require(p, req, model=self.name)


    def _sde_impl(self) -> Callable:
        """
        Returns the stepper for the SABR model with log-normal jumps.
        """
        p = self.params
        alpha, beta, rho = p["alpha"], p["beta"], p["rho"]
        lambda_, mu_j, sigma_j = p["lambda"], p["mu_j"], p["sigma_j"]
        rho_bar = np.sqrt(1 - rho**2)
        
        # Pre-calculate the risk-neutral jump compensator
        compensator = lambda_ * (np.exp(mu_j + 0.5 * sigma_j**2) - 1)

        # The stepper must accept the jump arguments provided by the MC engine
        def stepper(
            log_s_t: np.ndarray,
            v_t: np.ndarray, # SABR volatility sigma_t
            r: float,
            q: float,
            dt: float,
            dw_s: np.ndarray,
            dw_v: np.ndarray,
            jump_counts: np.ndarray,
            rng: np.random.Generator
        ) -> tuple[np.ndarray, np.ndarray]:
            
            # Evolve volatility (same as SABR)
            v_t_pos = np.abs(v_t)
            v_t_next = v_t_pos * np.exp(-0.5 * alpha**2 * dt + alpha * dw_v)
            
            # Evolve log-spot price (SABR drift + jump compensator)
            s_t = np.exp(log_s_t)
            # The drift term must account for the jumps
            s_drift = (r - q - compensator) * dt - 0.5 * (v_t_pos**2) * (s_t**(2 * beta - 2)) * dt
            s_diffusion = v_t_pos * (s_t**(beta - 1)) * (rho * dw_s + rho_bar * dw_v) # Note: dw_s here, not dw_v
            next_log_s = log_s_t + s_drift + s_diffusion
            
            # Add jumps (same as Bates)
            paths_with_jumps = np.where(jump_counts > 0)[0]
            for path_idx in paths_with_jumps:
                num_jumps = jump_counts[path_idx]
                jump_size = np.sum(rng.normal(loc=mu_j, scale=sigma_j, size=num_jumps))
                next_log_s[path_idx] += jump_size
            
            return next_log_s, v_t_next

        return stepper

    def _cf_impl(self, **kwargs: Any) -> CF:
        raise NotImplementedError(f"{self.name} does not have an analytic characteristic function.")
    def _pde_impl(self) -> Any:
        raise NotImplementedError(f"{self.name} PDE not implemented yet.")
    def _closed_form_impl(self, *args, **kwargs) -> Any:
        raise NotImplementedError(f"{self.name} does not have a closed-form solution.")