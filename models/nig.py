from __future__ import annotations
import numpy as np
from typing import Any, Callable

from models.base.base_model import BaseModel, CF
from models.base.validators import ParamValidator

class NIGModel(BaseModel):
    """
    Normal Inverse Gaussian (NIG) model. A pure-jump Lévy process.
    """
    name = "Normal Inverse Gaussian"
    supports_cf = True
    is_pure_levy = True

    def _validate_params(self) -> None:
        p = self.params
        # alpha: tail heaviness, beta: asymmetry, delta: scale
        req = ["alpha", "beta", "delta"]
        ParamValidator.require(p, req, model=self.name)
        ParamValidator.positive(p, ["alpha", "delta"], model=self.name)
        if not (abs(p["beta"]) < p["alpha"]):
            raise ValueError("NIG params must satisfy |beta| < alpha.")

    def _cf_impl(self, *, t: float, spot: float, r: float, q: float) -> CF:
        p = self.params
        alpha, beta, delta = p["alpha"], p["beta"], p["delta"]

        # Risk-neutral drift compensator
        compensator = delta * (np.sqrt(alpha**2 - (beta + 1)**2) - np.sqrt(alpha**2 - beta**2))
        drift = r - q + compensator

        def phi(u: np.ndarray | complex) -> np.ndarray | complex:
            term1 = 1j * u * (np.log(spot) + drift * t)
            term2 = delta * t * (np.sqrt(alpha**2 - beta**2) - np.sqrt(alpha**2 - (beta + 1j * u)**2))
            return np.exp(term1 + term2)

        return phi
    
    def sample_terminal_log_return(self, T: float, size: int, rng: np.random.Generator) -> np.ndarray:
        """
        Draws samples from the terminal distribution of the NIG process at time T.
        A NIG process is a Brownian motion with drift, time-changed by an
        Inverse Gaussian process.
        """
        from scipy.stats import invgauss # Local import
        p = self.params
        alpha, beta, delta = p["alpha"], p["beta"], p["delta"]
        
        # 1. Draw the time-change from an Inverse Gaussian distribution
        mu_ig = delta * T / np.sqrt(alpha**2 - beta**2)
        ig_time = invgauss.rvs(mu=mu_ig, size=size, random_state=rng)
        
        # 2. Draw from a Normal distribution with the IG-changed time
        bm_drift = p.get("mu", 0) * T + beta * ig_time
        bm_diffusion = np.sqrt(ig_time) * rng.standard_normal(size=size)
        
        return bm_drift + bm_diffusion

    def _pde_impl(self) -> Any:
        raise NotImplementedError(f"{self.name} PDE not implemented")
    
    def _closed_form_impl(self) -> Any:
        raise NotImplementedError(f"{self.name} Closed Form not implemented")

    def _sde_impl(self) -> Callable:
        # To be implemented later
        raise NotImplementedError(f"{self.name} SDE not implemented yet.")
    
    def raw_cf(self, t: float) -> Callable:
        """Returns the CF of the raw NIG process, without drift or spot."""
        p = self.params
        alpha, beta, delta = p["alpha"], p["beta"], p["delta"]
        def phi_raw(u: np.ndarray | complex) -> np.ndarray | complex:
            term = delta * (np.sqrt(alpha**2 - beta**2) - np.sqrt(alpha**2 - (beta + 1j * u)**2))
            return np.exp(t * term)
        return phi_raw