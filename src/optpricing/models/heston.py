from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from optpricing.models.base import CF, BaseModel, ParamValidator

__doc__ = """
Defines the Heston stochastic volatility model.
"""


class HestonModel(BaseModel):
    """
    Heston stochastic volatility model.

    This model describes the evolution of an asset's price where the volatility
    itself is a random process, following a Cox-Ingersoll-Ross (CIR) process.
    It is widely used as it can capture volatility smiles and skews.
    """

    name: str = "Heston"
    supports_cf: bool = True
    supports_sde: bool = True
    has_variance_process: bool = True
    cf_kwargs = ("v0",)  # v0 is required for pricing, passed from kwargs

    default_params = {
        "v0": 0.04,
        "kappa": 2.0,
        "theta": 0.04,
        "rho": -0.7,
        "vol_of_vol": 0.5,
    }
    param_defs = {
        "v0": {
            "label": "Initial Variance (v0)",
            "default": 0.04,
            "min": 0.001,
            "max": 0.5,
            "step": 0.01,
        },
        "kappa": {
            "label": "Mean Reversion",
            "default": 2.0,
            "min": 0.1,
            "max": 10.0,
            "step": 0.1,
        },
        "theta": {
            "label": "Long-Term Var",
            "default": 0.04,
            "min": 0.01,
            "max": 0.5,
            "step": 0.01,
        },
        "rho": {
            "label": "Correlation",
            "default": -0.7,
            "min": -0.99,
            "max": 0.99,
            "step": 0.05,
        },
        "vol_of_vol": {
            "label": "Vol of Vol",
            "default": 0.5,
            "min": 0.1,
            "max": 1.5,
            "step": 0.05,
        },
    }

    def __init__(self, params: dict[str, float] | None = None):
        """
        Initializes the Heston model.

        Parameters
        ----------
        params : dict[str, float] | None, optional
            A dictionary of model parameters. If None, `default_params` are used.
        """
        super().__init__(params or self.default_params)

    def _validate_params(self) -> None:
        """Validates parameters for mean-reversion, correlation, etc."""
        p = self.params
        req = ["kappa", "theta", "rho", "vol_of_vol"]
        ParamValidator.require(p, req, model=self.name)
        ParamValidator.positive(p, ["kappa", "theta", "vol_of_vol"], model=self.name)
        ParamValidator.bounded(p, "rho", -1.0, 1.0, model=self.name)
        if 2 * p["kappa"] * p["theta"] < p["vol_of_vol"] ** 2:
            print(
                f"Warning: Params for {self.name} do not satisfy the Feller condition."
            )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, HestonModel):
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
        v0: float,
        **_: Any,
    ) -> CF:
        """
        Heston characteristic function for the log-spot price log(S_t).

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
        v0 : float
            The initial variance of the asset's returns.

        Returns
        -------
        CF
            The characteristic function.
        """
        p = self.params
        kappa, theta, rho, vol_of_vol = (
            p["kappa"],
            p["theta"],
            p["rho"],
            p["vol_of_vol"],
        )

        def phi(u: np.ndarray | complex) -> np.ndarray | complex:
            d = np.sqrt(
                (rho * vol_of_vol * u * 1j - kappa) ** 2
                - (vol_of_vol**2) * (-u * 1j - u**2)
            )
            g = (kappa - rho * vol_of_vol * u * 1j - d) / (
                kappa - rho * vol_of_vol * u * 1j + d
            )
            C = (r - q) * u * 1j * t + (kappa * theta / vol_of_vol**2) * (
                (kappa - rho * vol_of_vol * u * 1j - d) * t
                - 2 * np.log((1 - g * np.exp(-d * t)) / (1 - g))
            )
            D = ((kappa - rho * vol_of_vol * u * 1j - d) / vol_of_vol**2) * (
                (1 - np.exp(-d * t)) / (1 - g * np.exp(-d * t))
            )
            return np.exp(C + D * v0 + 1j * u * np.log(spot))

        return phi

    def get_sde_stepper(self) -> Callable:
        """
        Returns the SDE stepper function for the Heston model.

        This is not used by the JIT-compiled kernel but is kept for potential
        future use with non-compiled or more complex simulation loops.
        """
        p = self.params
        kappa, theta, _, vol_of_vol = (
            p["kappa"],
            p["theta"],
            p["rho"],
            p["vol_of_vol"],
        )

        def stepper(
            log_s_t: np.ndarray,
            v_t: np.ndarray,
            r: float,
            q: float,
            dt: float,
            dw_s: np.ndarray,
            dw_v: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray]:
            v_t_pos = np.maximum(v_t, 0)
            v_sqrt = np.sqrt(v_t_pos)
            log_s_t_next = log_s_t + (r - q - 0.5 * v_t_pos) * dt + v_sqrt * dw_s
            v_t_next = v_t + kappa * (theta - v_t_pos) * dt + vol_of_vol * v_sqrt * dw_v
            return log_s_t_next, np.maximum(v_t_next, 0)  # Reflection scheme

        return stepper

    #  Abstract Method Implementations
    def _sde_impl(self, **kwargs: Any) -> Callable:
        return self.get_sde_stepper()

    def _pde_impl(self, **kwargs: Any) -> Any:
        raise NotImplementedError

    def _closed_form_impl(self, **kwargs: Any) -> Any:
        raise NotImplementedError
