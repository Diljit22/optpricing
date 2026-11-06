from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from optpricing.models.base import CF, BaseModel, ParamValidator

__doc__ = """
Defines the Normal Inverse Gaussian (NIG) model, a pure-jump Levy process.
"""


class NIGModel(BaseModel):
    """Normal Inverse Gaussian (NIG) model, a pure-jump Levy process."""

    name: str = "Normal Inverse Gaussian"
    supports_cf: bool = True
    is_pure_levy: bool = True

    default_params = {"alpha": 15.0, "beta": -5.0, "delta": 0.5}

    def __init__(self, params: dict[str, float] | None = None):
        """
        Initializes the NIG model.

        Parameters
        ----------
        params : dict[str, float] | None, optional
            A dictionary of model parameters. If None, `default_params` are used.
        """
        super().__init__(params or self.default_params)

    def _validate_params(self) -> None:
        p = self.params
        req = ["alpha", "beta", "delta"]
        ParamValidator.require(p, req, model=self.name)
        ParamValidator.positive(p, ["alpha", "delta"], model=self.name)
        if not (abs(p["beta"]) < p["alpha"]):
            raise ValueError("NIG params must satisfy |beta| < alpha.")
        if not (abs(p["beta"] + 1.0) < p["alpha"]):
            # The martingale correction uses sqrt(alpha^2 - (beta+1)^2), i.e. the
            # exponential moment E[exp(X_t)] must exist.
            raise ValueError(
                "NIG params must satisfy |beta + 1| < alpha for the risk-neutral "
                "measure to exist."
            )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, NIGModel):
            return NotImplemented
        return self.params == other.params

    def __hash__(self) -> int:
        return hash((self.__class__, tuple(sorted(self.params.items()))))

    def moment_strip(self) -> tuple[float, float]:
        """``|beta + s| < alpha`` (Barndorff-Nielsen 1997)."""
        alpha, beta = self.params["alpha"], self.params["beta"]
        return float(-alpha - beta), float(alpha - beta)

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
        Risk-neutral characteristic function for the log-spot price log(S_t).

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
        """
        Returns the CF of the raw NIG process, without drift or spot.

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
        self,
        T: float,
        size: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Generates exact samples of the terminal log-return for the NIG process.

        This leverages the representation of the NIG process as a Brownian
        motion with drift, time-changed by an Inverse Gaussian process.

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
        from scipy.stats import invgauss

        p = self.params
        alpha, beta, delta = p["alpha"], p["beta"], p["delta"]
        gamma = np.sqrt(alpha**2 - beta**2)
        # The NIG subordinator at horizon T is IG(mean = delta*T/gamma,
        # shape = (delta*T)^2) (Barndorff-Nielsen 1997). scipy parametrises
        # invgauss(mu, scale) as IG(mean = mu*scale, shape = scale).
        shape = (delta * T) ** 2
        ig_time = invgauss.rvs(
            mu=1.0 / (gamma * delta * T), scale=shape, size=size, random_state=rng
        )
        bm_drift = beta * ig_time
        bm_diffusion = np.sqrt(ig_time) * rng.standard_normal(size=size)
        return bm_drift + bm_diffusion

    #  Abstract Method Implementations
    def _sde_impl(self, **kwargs: Any) -> Any:
        raise NotImplementedError("NIG is a pure Levy process, use terminal sampling.")

    def _pde_impl(self, **kwargs: Any) -> Any:
        raise NotImplementedError

    def _closed_form_impl(self, **kwargs: Any) -> Any:
        raise NotImplementedError
