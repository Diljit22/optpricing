from __future__ import annotations
from typing import Any, Callable
import numpy as np
from models.base.base_model import BaseModel, CF
from models.base.validators import ParamValidator

class CEVModel(BaseModel):
    """
    Constant Elasticity of Variance (CEV) model.
    dS_t = r*S_t*dt + sigma*S_t^gamma*dW_t
    """
    name = "CEV"
    supports_sde = True

    def _validate_params(self) -> None:
        p = self.params
        # sigma: scale parameter, gamma: elasticity parameter
        req = ["sigma", "gamma"]
        ParamValidator.require(p, req, model=self.name)
        ParamValidator.positive(p, ["sigma"], model=self.name)

    def _sde_impl(self) -> Callable:
        """
        Returns a function that performs an EXACT simulation of the CEV process
        terminal value using the Non-Central Chi-Squared distribution.
        This avoids step-by-step Euler discretization error.
        
        This is a terminal sampler, but we hook it into the SDE interface
        for compatibility with the existing MC engine. It will be called once
        with dt=T.
        """
        from scipy.stats import ncx2
        p = self.params
        sigma, gamma = p["sigma"], p["gamma"]

        def exact_sampler(
            log_s_t: np.ndarray, # Initial log-spot
            r: float,
            q: float,
            T: float,
            dw_t: np.ndarray # Used for size and rng state only
        ) -> np.ndarray:
            
            S0 = np.exp(log_s_t)
            df = 2 + 2 / (1 - gamma)
            k = 2 * r / (sigma**2 * (1 - gamma) * (np.exp(r * (1 - gamma) * T) - 1))
            
            # The non-centrality parameter
            nc = k * S0**(2 * (1 - gamma)) * np.exp(-r * (1 - gamma) * T)
            
            # Draw from the non-central chi-squared distribution
            chi2_draws = ncx2.rvs(df=df, nc=nc, size=len(S0))
            
            ST = (chi2_draws / k)**(1 / (2 * (1 - gamma)))
            
            return np.log(ST)

        return exact_sampler

    def _cf_impl(self, **kwargs: Any) -> CF:
        raise NotImplementedError(f"{self.name} does not have an analytic characteristic function.")
    def _pde_impl(self) -> Any:
        raise NotImplementedError(f"{self.name} PDE not implemented yet.")
    def _closed_form_impl(self, *args, **kwargs) -> Any:
        raise NotImplementedError(f"{self.name} does not have a general closed-form solution.")