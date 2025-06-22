from __future__ import annotations

from typing import Any, Callable

import numpy as np

from quantfin.models.base import CF, BaseModel, ParamValidator


class NIGModel(BaseModel):
    """Normal Inverse Gaussian (NIG) model, a pure-jump Lévy process."""

    name: str = "Normal Inverse Gaussian"
    supports_cf: bool = True
    is_pure_levy: bool = True

    def _validate_params(self) -> None:
        p = self.params
        req = ["alpha", "beta", "delta"]
        ParamValidator.require(p, req, model=self.name)
        ParamValidator.positive(p, ["alpha", "delta"], model=self.name)
        if not (abs(p["beta"]) < p["alpha"]):
            raise ValueError("NIG params must satisfy |beta| < alpha.")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, NIGModel):
            return NotImplemented
        return self.params == other.params

    def __hash__(self) -> int:
        return hash((self.__class__, tuple(sorted(self.params.items()))))

    def _cf_impl(self, *, t: float, spot: float, r: float, q: float, **_: Any) -> CF:
        """Risk-neutral characteristic function for the log-spot price log(S_t)."""
        p = self.params
        alpha, beta, delta = p["alpha"], p["beta"], p["delta"]
        compensator = delta * (
            np.sqrt(alpha**2 - (beta + 1) ** 2) - np.sqrt(alpha**2 - beta**2)
        )
        drift = r - q + compensator

        def phi(u: np.ndarray | complex) -> np.ndarray | complex:
            term1 = 1j * u * (np.log(spot) + drift * t)
            term2 = (
                delta
                * t
                * (
                    np.sqrt(alpha**2 - beta**2)
                    - np.sqrt(alpha**2 - (beta + 1j * u) ** 2)
                )
            )
            return np.exp(term1 + term2)

        return phi

    def raw_cf(self, *, t: float) -> Callable:
        """Returns the CF of the raw NIG process, without drift or spot."""
        p = self.params
        alpha, beta, delta = p["alpha"], p["beta"], p["delta"]

        def phi_raw(u: np.ndarray | complex) -> np.ndarray | complex:
            term = (
                delta
                * t
                * (
                    np.sqrt(alpha**2 - beta**2)
                    - np.sqrt(alpha**2 - (beta + 1j * u) ** 2)
                )
            )
            return np.exp(term)

        return phi_raw

    def sample_terminal_log_return(
        self, T: float, size: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Draws samples from the terminal distribution of the raw NIG process."""
        from scipy.stats import invgauss

        p = self.params
        alpha, beta, delta = p["alpha"], p["beta"], p["delta"]
        mu_ig = delta * T / np.sqrt(alpha**2 - beta**2)
        ig_time = invgauss.rvs(mu=mu_ig, size=size, random_state=rng)
        bm_drift = beta * ig_time
        bm_diffusion = np.sqrt(ig_time) * rng.standard_normal(size=size)
        return bm_drift + bm_diffusion

    #  Abstract Method Implementations
    def _sde_impl(self, **kwargs: Any) -> Any:
        raise NotImplementedError("NIG is a pure Lévy process, use terminal sampling.")

    def _pde_impl(self, **kwargs: Any) -> Any:
        raise NotImplementedError

    def _closed_form_impl(self, **kwargs: Any) -> Any:
        raise NotImplementedError
