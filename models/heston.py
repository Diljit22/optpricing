from __future__ import annotations
import numpy as np
from typing import Any, Callable

from models.base.base_model import BaseModel, CF
from models.base.validators import ParamValidator

class HestonModel(BaseModel):
    """
    Heston (1993) stochastic volatility model.
    Provides a semi-closed-form characteristic function.
    """
    name = "Heston"
    supports_cf = True
    supports_sde = True
    has_variance_process = True
    cf_kwargs = BaseModel.cf_kwargs + ("v0",)

    def _validate_params(self) -> None:
        p = self.params
        # v0: initial variance, kappa: mean-reversion speed,
        # theta: long-run variance, rho: correlation, vol_of_vol: vol of variance
        req = ["kappa", "theta", "rho", "vol_of_vol"]
        ParamValidator.require(p, req, model=self.name)
        ParamValidator.positive(p, ["kappa", "theta", "vol_of_vol"], model=self.name)
        ParamValidator.bounded(p, "rho", -1.0, 1.0, model=self.name)
        # Feller condition: 2*kappa*theta >= vol_of_vol**2. Not strictly required but good practice.
        if 2 * p["kappa"] * p["theta"] < p["vol_of_vol"]**2:
            print(f"Warning: Heston parameters for {self.name} do not satisfy the Feller condition.")

    def _cf_impl(self, *, t: float, spot: float, r: float, q: float, v0: float) -> CF:
        """
        Heston characteristic function for log-spot price log(S_t).
        """
        p = self.params
        kappa, theta, rho, vol_of_vol = p["kappa"], p["theta"], p["rho"], p["vol_of_vol"]

        def phi(u: np.ndarray | complex) -> np.ndarray | complex:
            # Coefficients for the Riccati ODEs
            d = np.sqrt((rho * vol_of_vol * u * 1j - kappa)**2 - (vol_of_vol**2) * (-u * 1j - u**2))
            g = (kappa - rho * vol_of_vol * u * 1j - d) / (kappa - rho * vol_of_vol * u * 1j + d)

            # Solutions to the ODEs
            C = (r - q) * u * 1j * t + (kappa * theta / vol_of_vol**2) * (
                (kappa - rho * vol_of_vol * u * 1j - d) * t - 2 * np.log((1 - g * np.exp(-d * t)) / (1 - g))
            )
            D = ((kappa - rho * vol_of_vol * u * 1j - d) / vol_of_vol**2) * ((1 - np.exp(-d * t)) / (1 - g * np.exp(-d * t)))

            return np.exp(C + D * v0 + 1j * u * np.log(spot))

        return phi
    
    def _sde_impl(self) -> Callable:
        p = self.params
        kappa, theta, rho, vol_of_vol = p["kappa"], p["theta"], p["rho"], p["vol_of_vol"]
        rho_bar = np.sqrt(1 - rho**2)

        def stepper(
            log_s_t: np.ndarray,
            v_t: np.ndarray,
            r: float,
            q: float,
            dt: float,
            dw_s: np.ndarray,
            dw_v: np.ndarray
        ) -> tuple[np.ndarray, np.ndarray]:
            v_t_pos = np.maximum(v_t, 0)
            
            # Evolve variance (Reflection scheme to ensure positivity)
            v_t_next = v_t + kappa * (theta - v_t_pos) * dt + vol_of_vol * np.sqrt(v_t_pos) * dw_v
            v_t_next = np.abs(v_t_next)
            
            # Evolve log-spot price
            s_drift = (r - q - 0.5 * v_t_pos) * dt
            s_diffusion = np.sqrt(v_t_pos) * (rho * dw_v + rho_bar * dw_s)
            log_s_t_next = log_s_t + s_drift + s_diffusion
            
            return log_s_t_next, v_t_next

        return stepper

    def _pde_impl(self) -> Any:
        raise NotImplementedError(f"{self.name} PDE not implemented")

    def _closed_form_impl(self) -> Any:
        raise NotImplementedError(f"{self.name} Closed Form not implemented")
