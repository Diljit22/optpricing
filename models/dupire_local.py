from __future__ import annotations
from typing import Any, Callable
import numpy as np

from models.base.base_model import BaseModel, CF
from models.base.validators import ParamValidator

class DupireLocalVolModel(BaseModel):
    """
    Dupire (1994) Local Volatility model. The volatility is a deterministic
    function of time and spot price, sigma(t, S_t).
    """
    name = "Dupire Local Volatility"
    supports_sde = True
    
    def _validate_params(self) -> None:
        # This would validate that a valid surface object was passed in params.
        if "vol_surface" not in self.params:
            raise ValueError("Dupire model requires a 'vol_surface' in its parameters.")

    def _sde_impl(self) -> Callable:
        vol_surface = self.params["vol_surface"]

        def stepper(
            log_s_t: np.ndarray,
            r: float,
            q: float,
            dt: float,
            dw_t: np.ndarray,
            t_current: float
        ) -> np.ndarray:
            current_spot = np.exp(log_s_t)
            # Look up the local volatility from the surface for the current time and spot
            local_vol = vol_surface(t_current, current_spot)
            
            drift = (r - q - 0.5 * local_vol**2) * dt
            diffusion = local_vol * dw_t
            return log_s_t + drift + diffusion
            
        return stepper

    def _cf_impl(self, **kwargs: Any) -> CF:
        raise NotImplementedError(f"{self.name} does not have an analytic characteristic function.")
    def _pde_impl(self) -> Any:
        raise NotImplementedError(f"{self.name} PDE not implemented yet.")
    def _closed_form_impl(self, *args, **kwargs) -> Any:
        raise NotImplementedError(f"{self.name} does not have a closed-form solution.")