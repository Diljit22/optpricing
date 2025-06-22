from __future__ import annotations

from typing import Any, Callable

import numpy as np
from scipy.special import kv
from scipy.stats import genhyperbolic

from quantfin.models.base import CF, BaseModel, ParamValidator


class HyperbolicModel(BaseModel):
    """Hyperbolic pure-jump Lévy model."""

    name: str = "Hyperbolic"
    supports_cf: bool = True
    is_pure_levy: bool = True

    def _validate_params(self) -> None:
        p = self.params
        req = ["alpha", "beta", "delta", "mu"]
        ParamValidator.require(p, req, model=self.name)
        ParamValidator.positive(p, ["alpha", "delta"], model=self.name)
        if not (abs(p["beta"]) < p["alpha"]):
            raise ValueError("Hyperbolic params must satisfy |beta| < alpha.")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, HyperbolicModel):
            return NotImplemented
        return self.params == other.params

    def __hash__(self) -> int:
        return hash((self.__class__, tuple(sorted(self.params.items()))))

    def _cf_impl(
        self,
        *,
        t: float,
        spot: float,
        r: float,
        q: float,
        **_: Any,
    ) -> CF:
        p = self.params
        alpha, beta, delta, mu = p["alpha"], p["beta"], p["delta"], p["mu"]
        compensator = np.log(self.raw_cf(t=1.0)(-1j))  # E[exp(X_1)] = phi_raw(-i)
        drift = r - q - compensator

        def phi(u: np.ndarray | complex):
            return self.raw_cf(t=t)(u) * np.exp(1j * u * (np.log(spot) + drift * t))

        return phi

    def raw_cf(self, *, t: float) -> Callable:
        p = self.params
        alpha, beta, delta, mu = p["alpha"], p["beta"], p["delta"], p["mu"]
        gamma_0 = np.sqrt(alpha**2 - beta**2)

        def phi_raw(u: np.ndarray | complex) -> np.ndarray | complex:
            gamma_u = np.sqrt(alpha**2 - (beta + 1j * u) ** 2)
            term1 = np.exp(1j * u * mu * t)
            term2 = (gamma_0 / gamma_u) ** t * (
                kv(1, delta * t * gamma_u) / kv(1, delta * t * gamma_0)
            )
            return term1 * term2

        return phi_raw

    def sample_terminal_log_return(
        self,
        T: float,
        size: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        p = self.params
        return genhyperbolic.rvs(
            p=1.0,
            a=p["alpha"],
            b=p["beta"],
            loc=p["mu"] * T,
            scale=p["delta"] * T,
            size=size,
            random_state=rng,
        )

    #  Abstract Method Implementations
    def _sde_impl(self, **kwargs: Any) -> Any:
        raise NotImplementedError

    def _pde_impl(self, **kwargs: Any) -> Any:
        raise NotImplementedError

    def _closed_form_impl(self, **kwargs: Any) -> Any:
        raise NotImplementedError
