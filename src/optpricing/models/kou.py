from __future__ import annotations

from typing import Any

import numpy as np

from optpricing.models.base import CF, BaseModel, ParamValidator

__doc__ = """
Defines the Kou double-exponential jump-diffusion model.
"""


class KouModel(BaseModel):
    """Kou double-exponential jump-diffusion model."""

    name: str = "Kou Double-Exponential Jump"
    supports_cf: bool = True
    supports_sde: bool = True
    has_jumps: bool = True

    default_params = {
        "sigma": 0.15,
        "lambda": 1.0,
        "p_up": 0.6,
        "eta1": 10.0,
        "eta2": 5.0,
    }
    param_defs = {
        "sigma": {
            "label": "Volatility",
            "default": 0.15,
            "min": 0.01,
            "max": 1.0,
            "step": 0.01,
        },
        "lambda": {
            "label": "Jump Intensity",
            "default": 1.0,
            "min": 0.0,
            "max": 10.0,
            "step": 0.1,
        },
        "p_up": {
            "label": "Up Prob",
            "default": 0.6,
            "min": 0.0,
            "max": 1.0,
            "step": 0.05,
        },
        "eta1": {
            "label": "Up Size",
            "default": 10.0,
            "min": 0.1,
            "max": 50.0,
            "step": 0.5,
        },
        "eta2": {
            "label": "Down Size",
            "default": 5.0,
            "min": 0.1,
            "max": 50.0,
            "step": 0.5,
        },
    }

    def __init__(self, params: dict[str, float] | None = None):
        """
        Initializes the Kou Double-Exponential Jump model.

        Parameters
        ----------
        params : dict[str, float] | None, optional
            A dictionary of model parameters. If None, `default_params` are used.
        """
        super().__init__(params or self.default_params)

    def _validate_params(self) -> None:
        """Validate the models parameter."""
        p = self.params
        req = ["sigma", "lambda", "p_up", "eta1", "eta2"]
        ParamValidator.require(p, req, model=self.name)
        ParamValidator.positive(p, ["sigma", "lambda", "eta1", "eta2"], model=self.name)
        ParamValidator.bounded(p, "p_up", 0.0, 1.0, model=self.name)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, KouModel):
            return NotImplemented
        return self.params == other.params

    def __hash__(self) -> int:
        return hash((self.__class__, tuple(sorted(self.params.items()))))

    def _cf_impl(self, *, t: float, spot: float, r: float, q: float, **_: Any) -> CF:
        """
        Kou characteristic function for the log-spot price log(S_t).

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
        sigma, lambda_, p_up, eta1, eta2 = (
            p["sigma"],
            p["lambda"],
            p["p_up"],
            p["eta1"],
            p["eta2"],
        )

        compensator = lambda_ * (
            (p_up * eta1 / (eta1 - 1)) + ((1 - p_up) * eta2 / (eta2 + 1)) - 1
        )
        drift = r - q - 0.5 * sigma**2 - compensator

        def phi(u: np.ndarray | complex) -> np.ndarray | complex:
            # BSM component
            bsm_part = 1j * u * (np.log(spot) + drift * t) - 0.5 * u**2 * sigma**2 * t
            # Jump component
            phi_jump = (p_up * eta1 / (eta1 - 1j * u)) + (
                (1 - p_up) * eta2 / (eta2 + 1j * u)
            )
            jump_part = lambda_ * t * (phi_jump - 1)

            return np.exp(bsm_part + jump_part)

        return phi

    #  Abstract Method Implementations
    def _sde_impl(self, **kwargs: Any) -> Any:
        raise NotImplementedError("Kou uses a specialized kernel.")

    def _pde_impl(self, **kwargs: Any) -> Any:
        raise NotImplementedError

    def _closed_form_impl(self, **kwargs: Any) -> Any:
        raise NotImplementedError
