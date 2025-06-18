from __future__ import annotations
import numpy as np
from scipy.special import kv
from typing import Any, Callable
from scipy.stats import genhyperbolic

from quantfin.models.base import BaseModel, ParamValidator, CF


class HyperbolicModel(BaseModel):
    """
    Hyperbolic pure-jump Lévy model.
    """
    name = "Hyperbolic"
    supports_cf = True
    is_pure_levy = True

    def _validate_params(self) -> None:
        p = self.params
        req = ["alpha", "beta", "delta", "mu"]
        ParamValidator.require(p, req, model=self.name)
        ParamValidator.positive(p, ["alpha", "delta"], model=self.name)
        if not (abs(p["beta"]) < p["alpha"]):
            raise ValueError("Hyperbolic params must satisfy |beta| < alpha.")

    def _cf_impl(self, *, t: float, spot: float, r: float, q: float) -> CF:
        p = self.params
        alpha, beta, delta, mu = p["alpha"], p["beta"], p["delta"], p["mu"]

        gamma_0 = np.sqrt(alpha**2 - beta**2)     
        gamma_1 = np.sqrt(alpha**2 - (beta + 1)**2)      

        # real arguments -> float is fine
        K0 = kv(1, delta*gamma_0) 
        K1 = kv(1, delta*gamma_1) 
        if K0 == 0:                           
            raise ValueError("Hyperbolic parameters give K1 ~0 → compensator blow-up")

        compensator = np.exp(mu) * (gamma_0*K1) / (gamma_1*K0) - 1
        drift = r - q - compensator

        def phi(u: np.ndarray | complex):
            gamma_u = np.sqrt(alpha**2 - (beta + 1j*u)**2)         # complex array/scalar
            term1 = 1j*u*(np.log(spot) + drift*t)
            term2 = 1j*u*mu*t
            term3 = (gamma_0/gamma_u)**t * kv(1, delta*gamma_u*t)/kv(1, delta*gamma_0*t)
            return np.exp(term1) * np.exp(term2) * term3

        return phi

    def _sde_impl(self) -> Any:
        raise NotImplementedError(f"{self.name} SDE not implemented yet.")
    def _pde_impl(self) -> Any:
        raise NotImplementedError(f"{self.name} PDE not implemented yet.")
    def _closed_form_impl(self, *args, **kwargs) -> Any:
        raise NotImplementedError(f"{self.name} does not have a closed-form solution.")
    
    def raw_cf(self, t: float) -> Callable[[np.ndarray | complex], np.ndarray | complex]:
        """
        CF of the *raw* Hyperbolic Lévy process (no spot, no drift adjustment).
        """
        p = self.params
        alpha, beta, delta = p["alpha"], p["beta"], p["delta"]

        gamma_0 = np.sqrt(alpha**2 - beta**2)                    # real
        def phi_raw(u: np.ndarray | complex) -> np.ndarray | complex:
            gamma_u = np.sqrt(alpha**2 - (beta + 1j * u) ** 2)   # complex
            # lam = 1 for the Hyperbolic subclass of GH
            return (gamma_0 / gamma_u) ** t * (
                kv(1, delta * gamma_u * t) / kv(1, delta * gamma_0 * t)
            )

        return phi_raw

    def sample_terminal_log_return(
        self, T: float, size: int, rng: np.random.Generator
    ) -> np.ndarray:
        """
        Generates i.i.d. samples of X_T  (the log-return over horizon T).

        Uses SciPy's `genhyperbolic`, which draws directly from the
        Generalised-Hyperbolic law.  For the *Hyperbolic* subclass we set lam = 1.
        """
        p = self.params
        alpha, beta, delta, mu = (
            p["alpha"],
            p["beta"],
            p["delta"],
            p["mu"],
        )

        scale = delta * T
        loc   = mu * T            # deterministic drift piece

        return genhyperbolic.rvs(
            p=1.0,               # lam = 1   (Hyperbolic)
            a=alpha,
            b=beta,
            loc=loc,
            scale=scale,
            size=size,
            random_state=rng,
        )