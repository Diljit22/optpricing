from __future__ import annotations
import numpy as np
from typing import Any, Callable

from models.base.base_model import BaseModel, CF
from models.base.validators import ParamValidator

class VarianceGammaModel(BaseModel):
    """
    Variance Gamma (VG) model. A pure-jump Lévy process.
    """
    name = "Variance Gamma"
    supports_cf = True
    is_pure_levy = True

    def _validate_params(self) -> None:
        p = self.params
        # sigma: volatility of the BM, nu: variance rate of the gamma process
        # theta: drift of the BM
        req = ["sigma", "nu", "theta"]
        ParamValidator.require(p, req, model=self.name)
        ParamValidator.positive(p, ["sigma", "nu"], model=self.name)

    def _cf_impl(self, *, t: float, spot: float, r: float, q: float) -> CF:
        p = self.params
        sigma, nu, theta = p["sigma"], p["nu"], p["theta"]

        # This term can be complex if nu or theta are, but for now it's real.
        # The log can still fail if the argument is negative.
        log_arg = 1 - theta * nu - 0.5 * sigma**2 * nu
        if log_arg <= 0:
            raise ValueError("VG parameters do not allow for a finite price (moment condition not met).")
        
        compensator = (1/nu) * np.log(log_arg)
        drift = r - q - compensator

        def phi(u: np.ndarray | complex) -> np.ndarray | complex:
            # This term will be complex for complex u, so np.log is essential.
            log_of_term = np.log(1 - 1j * u * theta * nu + 0.5 * u**2 * sigma**2 * nu)
            
            term1 = 1j * u * (np.log(spot) + drift * t)
            term2 = (-t / nu) * log_of_term
            return np.exp(term1 + term2)

        return phi
    
    def sample_terminal_log_return(self, T: float, size: int, rng: np.random.Generator) -> np.ndarray:
        """
        Draws samples from the terminal distribution of the VG process at time T.
        A VG process is a Brownian motion with drift, time-changed by a Gamma process.
        """
        p = self.params
        sigma, nu, theta = p["sigma"], p["nu"], p["theta"]
        
        # 1. Draw the time-change from a Gamma distribution
        gamma_time = rng.gamma(shape=T/nu, scale=nu, size=size)
        
        # 2. Draw from a Normal distribution with the gamma-changed time
        bm_drift = theta * gamma_time
        bm_diffusion = sigma * np.sqrt(gamma_time) * rng.standard_normal(size=size)
        
        return bm_drift + bm_diffusion

    def _pde_impl(self) -> Any:
        raise NotImplementedError(f"{self.name} PDE not implemented")
    
    def _closed_form_impl(self) -> Any:
        raise NotImplementedError(f"{self.name} Closed Form not implemented")

    def _sde_impl(self) -> Callable:
        # To be implemented later
        raise NotImplementedError(f"{self.name} SDE not implemented yet.")
    
# In class VarianceGammaModel
    def raw_cf(self, t: float) -> Callable:
        """Returns the CF of the raw VG process, without drift or spot."""
        p = self.params
        sigma, nu, theta = p["sigma"], p["nu"], p["theta"]
        def phi_raw(u: np.ndarray | complex) -> np.ndarray | complex:
            return (1 - 1j * u * theta * nu + 0.5 * u**2 * sigma**2 * nu)**(-t / nu)
        return phi_raw