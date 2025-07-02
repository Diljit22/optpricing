from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from optpricing.models.base import CF, BaseModel, ParamValidator
from optpricing.models.heston import HestonModel
from optpricing.models.merton_jump import MertonJumpModel

__doc__ = """
Defines the Bates stochastic volatility with jumps model.
"""


class BatesModel(BaseModel):
    """
    Bates stochastic volatility jump-diffusion model.

    This model combines the Heston stochastic volatility process with Merton's
    log-normal jump-diffusion process, providing a rich framework for capturing
    both volatility smiles and sudden market shocks.
    """

    name: str = "Bates"
    supports_cf: bool = True
    supports_sde: bool = True
    has_jumps: bool = True
    has_variance_process: bool = True
    cf_kwargs = ("v0",)

    default_params = {
        "v0": 0.04,
        "kappa": 2.0,
        "theta": 0.04,
        "rho": -0.7,
        "vol_of_vol": 0.5,
        "lambda": 0.5,
        "mu_j": -0.1,
        "sigma_j": 0.15,
    }
    param_defs = {**HestonModel.param_defs, **MertonJumpModel.param_defs}

    def __init__(self, params: dict[str, float] | None = None):
        """
        Initializes the Bates model.

        Parameters
        ----------
        params : dict[str, float] | None, optional
            A dictionary of model parameters. If None, `default_params` are used.
        """
        super().__init__(params or self.default_params)

    def _validate_params(self) -> None:
        """Validates all Heston and Merton parameters."""
        p = self.params
        req = ["kappa", "theta", "rho", "vol_of_vol", "lambda", "mu_j", "sigma_j"]
        ParamValidator.require(p, req, model=self.name)
        ParamValidator.positive(
            p, ["kappa", "theta", "vol_of_vol", "lambda", "sigma_j"], model=self.name
        )
        ParamValidator.bounded(p, "rho", -1.0, 1.0, model=self.name)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BatesModel):
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
        Bates characteristic function, a product of Heston and Merton CFs.

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
        lambda_, mu_j, sigma_j = p["lambda"], p["mu_j"], p["sigma_j"]

        # Risk-neutral compensator for the jump part
        compensator = lambda_ * (np.exp(mu_j + 0.5 * sigma_j**2) - 1)

        def phi(u: np.ndarray | complex) -> np.ndarray | complex:
            # Heston part (with drift adjusted for jump compensator)
            d = np.sqrt(
                (rho * vol_of_vol * u * 1j - kappa) ** 2
                - (vol_of_vol**2) * (-u * 1j - u**2)
            )
            g = (kappa - rho * vol_of_vol * u * 1j - d) / (
                kappa - rho * vol_of_vol * u * 1j + d
            )
            C = (r - q - compensator) * u * 1j * t + (kappa * theta / vol_of_vol**2) * (
                (kappa - rho * vol_of_vol * u * 1j - d) * t
                - 2 * np.log((1 - g * np.exp(-d * t)) / (1 - g))
            )
            D = ((kappa - rho * vol_of_vol * u * 1j - d) / vol_of_vol**2) * (
                (1 - np.exp(-d * t)) / (1 - g * np.exp(-d * t))
            )
            heston_part = np.exp(C + D * v0 + 1j * u * np.log(spot))

            # Merton jump part
            jump_part = np.exp(
                lambda_ * t * (np.exp(1j * u * mu_j - 0.5 * u**2 * sigma_j**2) - 1)
            )

            return heston_part * jump_part

        return phi

    def get_sde_stepper(self) -> Callable:
        """Returns the SDE stepper function for the Bates model."""
        p = self.params
        kappa, theta, _, vol_of_vol = (
            p["kappa"],
            p["theta"],
            p["rho"],
            p["vol_of_vol"],
        )
        lambda_, mu_j, sigma_j = p["lambda"], p["mu_j"], p["sigma_j"]
        compensator = lambda_ * (np.exp(mu_j + 0.5 * sigma_j**2) - 1)

        def stepper(
            log_s_t: np.ndarray,
            v_t: np.ndarray,
            r: float,
            q: float,
            dt: float,
            dw_s: np.ndarray,
            dw_v: np.ndarray,
            jump_counts: np.ndarray,
            rng: np.random.Generator,
        ) -> tuple[np.ndarray, np.ndarray]:
            v_t_pos = np.maximum(v_t, 0)
            v_sqrt = np.sqrt(v_t_pos)
            # Evolve variance
            v_t_next = v_t + kappa * (theta - v_t_pos) * dt + vol_of_vol * v_sqrt * dw_v
            # Evolve log-spot (Heston part + jump compensator)
            s_drift = (r - q - 0.5 * v_t_pos - compensator) * dt
            next_log_s = log_s_t + s_drift + v_sqrt * dw_s
            # Add jumps
            paths_with_jumps = np.where(jump_counts > 0)[0]
            for path_idx in paths_with_jumps:
                num_jumps = jump_counts[path_idx]
                jump_size = np.sum(rng.normal(loc=mu_j, scale=sigma_j, size=num_jumps))
                next_log_s[path_idx] += jump_size
            return next_log_s, np.maximum(v_t_next, 0)

        return stepper

    #  Abstract Method Implementations
    def _sde_impl(self, **kwargs: Any) -> Callable:
        return self.get_sde_stepper()

    def _pde_impl(self, **kwargs: Any) -> Any:
        raise NotImplementedError

    def _closed_form_impl(self, **kwargs: Any) -> Any:
        raise NotImplementedError
