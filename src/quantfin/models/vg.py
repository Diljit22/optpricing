from __future__ import annotations

from typing import Any, Callable

import numpy as np

from quantfin.models.base import CF, BaseModel, ParamValidator


class VarianceGammaModel(BaseModel):
    """Variance Gamma (VG) model, a pure-jump Lévy process."""

    name: str = "Variance Gamma"
    supports_cf: bool = True
    is_pure_levy: bool = True

    default_params = {"sigma": 0.2, "nu": 0.1, "theta": -0.14}
    param_defs = {
        "sigma": {
            "label": "Volatility (σ)",
            "default": 0.2,
            "min": 0.01,
            "max": 2.0,
            "step": 0.01,
        },
        "nu": {
            "label": "Variance Rate (ν)",
            "default": 0.1,
            "min": 0.001,
            "max": 2.0,
            "step": 0.01,
        },
        "theta": {
            "label": "Drift (θ)",
            "default": -0.14,
            "min": -2.0,
            "max": 2.0,
            "step": 0.05,
        },
    }

    def __init__(self, params: dict[str, float] | None = None):
        super().__init__(params or self.default_params)

    def _validate_params(self) -> None:
        p = self.params
        req = ["sigma", "nu", "theta"]
        ParamValidator.require(p, req, model=self.name)
        ParamValidator.positive(p, ["sigma", "nu"], model=self.name)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, VarianceGammaModel):
            return NotImplemented
        return self.params == other.params

    def __hash__(self) -> int:
        return hash((self.__class__, tuple(sorted(self.params.items()))))

    def _cf_impl(self, *, t: float, spot: float, r: float, q: float, **_: Any) -> CF:
        compensator = np.log(self.raw_cf(t=1.0)(-1j))
        drift = r - q - compensator

        def phi(u: np.ndarray | complex) -> np.ndarray | complex:
            return self.raw_cf(t=t)(u) * np.exp(1j * u * (np.log(spot) + drift * t))

        return phi

    def raw_cf(self, *, t: float) -> Callable:
        p = self.params
        sigma, nu, theta = p["sigma"], p["nu"], p["theta"]

        def phi_raw(u: np.ndarray | complex) -> np.ndarray | complex:
            return (1 - 1j * u * theta * nu + 0.5 * u**2 * sigma**2 * nu) ** (-t / nu)

        return phi_raw

    def sample_terminal_log_return(
        self, T: float, size: int, rng: np.random.Generator
    ) -> np.ndarray:
        p = self.params
        sigma, nu, theta = p["sigma"], p["nu"], p["theta"]
        gamma_time = rng.gamma(shape=T / nu, scale=nu, size=size)
        bm_drift = theta * gamma_time
        bm_diffusion = sigma * np.sqrt(gamma_time) * rng.standard_normal(size=size)
        return bm_drift + bm_diffusion

    #  Abstract Method Implementations
    def _sde_impl(self, **kwargs: Any) -> Any:
        raise NotImplementedError

    def _pde_impl(self, **kwargs: Any) -> Any:
        raise NotImplementedError

    def _closed_form_impl(self, **kwargs: Any) -> Any:
        raise NotImplementedError
