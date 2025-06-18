from __future__ import annotations
import numpy as np
from typing import Any, Callable

from quantfin.models.base import BaseModel, ParamValidator, CF


class KouModel(BaseModel):
    """
    Kou (2002) double-exponential jump-diffusion model.
    """
    name = "Kou Double-Exponential Jump"
    supports_cf = True
    supports_sde = True
    has_jumps = True

    def _validate_params(self) -> None:
        p = self.params
        # p_up: prob of positive jump, eta1: positive jump size, eta2: negative jump size
        req = ["sigma", "lambda", "p_up", "eta1", "eta2"]
        ParamValidator.require(p, req, model=self.name)
        ParamValidator.positive(p, ["sigma", "lambda", "eta1", "eta2"], model=self.name)
        ParamValidator.bounded(p, "p_up", 0.0, 1.0, model=self.name)

    def _cf_impl(self, *, t: float, spot: float, r: float, q: float) -> CF:
        """
        Kou characteristic function for log-spot price log(S_t).
        """
        p = self.params
        sigma, lambda_, p_up, eta1, eta2 = p["sigma"], p["lambda"], p["p_up"], p["eta1"], p["eta2"]

        # Risk-neutral drift compensator
        compensator = lambda_ * ((p_up * eta1 / (eta1 - 1)) + ((1 - p_up) * eta2 / (eta2 + 1)) - 1)
        drift = r - q - 0.5 * sigma**2 - compensator

        def phi(u: np.ndarray | complex) -> np.ndarray | complex:
            # BSM component
            bsm_part = 1j * u * (np.log(spot) + drift * t) - 0.5 * u**2 * sigma**2 * t
            # Jump component
            jump_part = 1j * u * lambda_ * t * (p_up / (eta1 - 1j * u) - (1 - p_up) / (eta2 + 1j * u))
            
            return np.exp(bsm_part + jump_part)

        return phi
    
    def _sde_impl(self) -> Callable:
        p = self.params
        sigma, p_up, eta1, eta2 = p["sigma"], p["p_up"], p["eta1"], p["eta2"]
        lambda_ = p["lambda"]

        # The risk-neutral compensator for the jump part is constant
        compensator = lambda_ * ((p_up * eta1 / (eta1 - 1)) + ((1 - p_up) * eta2 / (eta2 + 1)) - 1)

        def stepper(
            log_s_t: np.ndarray,
            r: float,
            q: float,
            dt: float,
            dw_t: np.ndarray,
            jump_counts: np.ndarray,
            rng: np.random.Generator
        ) -> np.ndarray:
            
            # Calculate the full drift term inside the stepper where r and q are defined
            drift_term = (r - q - 0.5 * sigma**2 - compensator) * dt

            # 1. Evolve with the continuous diffusion part first
            next_log_s = log_s_t + drift_term + sigma * dw_t
            
            # 2. Add the jumps for paths where they occurred
            paths_with_jumps = np.where(jump_counts > 0)[0]
            
            for path_idx in paths_with_jumps:
                num_jumps_on_path = jump_counts[path_idx]
                
                is_up_jump = rng.random(size=num_jumps_on_path) < p_up
                num_up_jumps = np.sum(is_up_jump)
                num_down_jumps = num_jumps_on_path - num_up_jumps
                
                total_jump_size = 0
                if num_up_jumps > 0:
                    total_jump_size += np.sum(rng.exponential(scale=1/eta1, size=num_up_jumps))
                if num_down_jumps > 0:
                    total_jump_size -= np.sum(rng.exponential(scale=1/eta2, size=num_down_jumps))
                
                next_log_s[path_idx] += total_jump_size

            return next_log_s

        return stepper

    def _pde_impl(self) -> Any:
        raise NotImplementedError(f"{self.name} PDE not implemented")

    def _closed_form_impl(self) -> Any:
        raise NotImplementedError(f"{self.name} Closed Form not implemented")