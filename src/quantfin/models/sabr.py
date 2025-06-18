from __future__ import annotations
from typing import Any, Callable
import numpy as np

from quantfin.models.base import BaseModel, ParamValidator, CF


class SABRModel(BaseModel):
    """
    Stochastic Alpha, Beta, Rho (SABR) model from Hagan et al. (2002).
    This is a stochastic volatility model widely used for interest rate derivatives.
    """
    name = "SABR"
    supports_sde = True
    has_variance_process = True # Flag for the MC dispatcher
    # No analytic CF

    def _validate_params(self) -> None:
        p = self.params
        # alpha: vol-of-vol, beta: CEV exponent, rho: correlation, nu: alias for vol-of-vol
        req = ["alpha", "beta", "rho"]
        ParamValidator.require(p, req, model=self.name)
        ParamValidator.positive(p, ["alpha"], model=self.name)
        ParamValidator.bounded(p, "beta", 0.0, 1.0, model=self.name)
        ParamValidator.bounded(p, "rho", -1.0, 1.0, model=self.name)

    def _sde_impl(self) -> Callable:
        """
        Returns the Euler-Maruyama stepper for the SABR model.
        Uses a reflection scheme for the volatility process.
        """
        p = self.params
        alpha, beta, rho = p["alpha"], p["beta"], p["rho"]
        rho_bar = np.sqrt(1 - rho**2)

        def stepper(
            log_s_t: np.ndarray,
            v_t: np.ndarray, # Here, 'v_t' represents the stochastic volatility sigma_t
            r: float,
            q: float,
            dt: float,
            dw_s: np.ndarray,
            dw_v: np.ndarray
        ) -> tuple[np.ndarray, np.ndarray]:
            
            # Evolve volatility (sigma_t) - it's a lognormal process
            # Use absolute value to ensure it stays positive (reflection)
            v_t_pos = np.abs(v_t)
            v_t_next = v_t_pos * np.exp(-0.5 * alpha**2 * dt + alpha * dw_v)
            
            # Evolve log-spot price
            s_t = np.exp(log_s_t)
            s_drift = (r - q) * dt - 0.5 * (v_t_pos**2) * (s_t**(2 * beta - 2)) * dt
            s_diffusion = v_t_pos * (s_t**(beta - 1)) * (rho * dw_v + rho_bar * dw_s)
            
            log_s_t_next = log_s_t + s_drift + s_diffusion
            
            return log_s_t_next, v_t_next

        return stepper

    def _cf_impl(self, **kwargs: Any) -> CF:
        raise NotImplementedError(f"{self.name} does not have an analytic characteristic function.")
    def _pde_impl(self) -> Any:
        raise NotImplementedError(f"{self.name} PDE not implemented yet.")
    def _closed_form_impl(self, *args, **kwargs) -> Any:
        # Note: There is a famous "Hagan's formula" which is an *approximation*, not a true closed form.
        raise NotImplementedError(f"{self.name} does not have a true closed-form solution.")