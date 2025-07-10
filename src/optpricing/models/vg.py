from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from optpricing.models.base import CF, BaseModel, ParamValidator

__doc__ = """
Defines the Variance Gamma (VG) model, a pure-jump Levy process.
"""


class VarianceGammaModel(BaseModel):
    """Variance Gamma (VG) model, a pure-jump Levy process."""

    name: str = "Variance Gamma"
    supports_cf: bool = True
    is_pure_levy: bool = True

    default_params = {"sigma": 0.2, "nu": 0.1, "theta": -0.14}
    param_defs = {
        "sigma": {
            "label": "Volatility",
            "default": 0.2,
            "min": 0.01,
            "max": 2.0,
            "step": 0.01,
        },
        "nu": {
            "label": "Variance Rate",
            "default": 0.1,
            "min": 0.001,
            "max": 2.0,
            "step": 0.01,
        },
        "theta": {
            "label": "Drift",
            "default": -0.14,
            "min": -2.0,
            "max": 2.0,
            "step": 0.05,
        },
    }

    def __init__(self, params: dict[str, float] | None = None):
        """
        Initializes the Variance Gamma (VG) model.

        Parameters
        ----------
        params : dict[str, float]
            A dictionary of model parameters.
        """
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

    def _cf_impl(
        self,
        *,
        t: float,
        spot: float,
        r: float,
        q: float,
        **_: Any,
    ) -> CF:
        """
        Variance Gamma characteristic function for the log-spot price log(S_t).

        This is the risk-neutral characteristic function, which includes the
        drift adjustment.

        Parameters
        ----------
        t : float
            The time to maturity of the option, in years.
        spot : float
            The current price of the underlying asset.
        r : float
            The continuously compounded risk-free rate.
        q : float
            The continuously compounded dividend yield.

        Returns
        -------
        CF
            The characteristic function.
        """
        compensator = np.log(self.raw_cf(t=1.0)(-1j))
        drift = r - q - compensator

        def phi(u: np.ndarray | complex) -> np.ndarray | complex:
            return self.raw_cf(t=t)(u) * np.exp(1j * u * (np.log(spot) + drift * t))

        return phi

    def raw_cf(self, *, t: float) -> Callable:
        """
        Returns the raw characteristic function of the VG process itself.

        This function represents the characteristic function of the process
        before the risk-neutral drift adjustment is applied.

        Parameters
        ----------
        t : float
            The time horizon.

        Returns
        -------
        Callable
            The raw characteristic function.
        """
        p = self.params
        sigma, nu, theta = p["sigma"], p["nu"], p["theta"]

        def phi_raw(u: np.ndarray | complex) -> np.ndarray | complex:
            return (1 - 1j * u * theta * nu + 0.5 * u**2 * sigma**2 * nu) ** (-t / nu)

        return phi_raw

    def sample_terminal_log_return(
        self,
        T: float,
        size: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Generates exact samples of the terminal log-return for the VG process.

        This leverages the representation of the VG process as a Brownian
        motion with drift, time-changed by a Gamma process.

        Parameters
        ----------
        T : float
            The time to maturity, in years.
        size : int
            The number of samples to generate.
        rng : np.random.Generator
            A random number generator instance for reproducibility.

        Returns
        -------
        np.ndarray
            An array of simulated terminal log-returns.
        """

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
