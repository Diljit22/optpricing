from __future__ import annotations

from typing import Any

import numpy as np

from optpricing.models.base import BaseModel, ParamValidator

__doc__ = """
Defines the Constant Elasticity of Variance (CEV) model.
"""


class CEVModel(BaseModel):
    """Constant Elasticity of Variance (CEV) model."""

    name: str = "CEV"
    supports_sde: bool = True
    has_exact_sampler: bool = True
    default_params = {"sigma": 0.8, "gamma": 0.7}

    def __init__(self, params: dict[str, float] | None = None):
        """
        Initializes the CEV model.

        Parameters
        ----------
        params : dict[str, float] | None, optional
            A dictionary of model parameters. If None, `default_params` are used.
            Must contain 'sigma' and 'gamma'. Defaults to None.
        """
        super().__init__(params or self.default_params)

    def _validate_params(self) -> None:
        """Validates that 'sigma' and 'gamma' are present and positive."""
        p = self.params
        req = ["sigma", "gamma"]
        ParamValidator.require(p, req, model=self.name)
        ParamValidator.positive(p, ["sigma"], model=self.name)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CEVModel):
            return NotImplemented
        return self.params == other.params

    def __hash__(self) -> int:
        return hash((self.__class__, tuple(sorted(self.params.items()))))

    def sample_terminal_spot(
        self,
        S0: float,
        r: float,
        T: float,
        size: int,
    ) -> np.ndarray:
        """
        Exact simulation of the terminal spot price via Non-Central Chi-Squared.

        This method provides a way to sample directly from the terminal
        distribution of the CEV process, which is more efficient than
        path-based simulation for pricing European options.

        Parameters
        ----------
        S0 : float
            The initial spot price of the asset.
        r : float
            The continuously compounded risk-free rate.
        T : float
            The time to maturity, in years.
        size : int
            The number of samples to generate.

        Returns
        -------
        np.ndarray
            An array of simulated terminal spot prices.
        """
        from scipy.stats import ncx2

        p = self.params
        sigma, gamma = p["sigma"], p["gamma"]

        # Handle the boundary case where gamma -> 1 (which is BSM)
        if abs(1.0 - gamma) < 1e-6:
            drift = (r - 0.5 * sigma**2) * T
            diffusion = sigma * np.sqrt(T) * np.random.standard_normal(size)  # noqa: NPY002
            return S0 * np.exp(drift + diffusion)

        df = 2 + 2 / (1 - gamma)
        k = 2 * r / (sigma**2 * (1 - gamma) * (np.exp(r * (1 - gamma) * T) - 1))
        nc = k * S0 ** (2 * (1 - gamma)) * np.exp(-r * (1 - gamma) * T)

        chi2_draws = ncx2.rvs(df=df, nc=nc, size=size)
        ST = (chi2_draws / k) ** (1 / (2 * (1 - gamma)))
        return ST

    #  Abstract Method Implementations
    def _sde_impl(self, **kwargs: Any) -> Any:
        raise NotImplementedError("CEV uses a specialized exact sampler.")

    def _cf_impl(self, **kwargs: Any) -> Any:
        raise NotImplementedError

    def _pde_impl(self, **kwargs: Any) -> Any:
        raise NotImplementedError

    def _closed_form_impl(self, **kwargs: Any) -> Any:
        raise NotImplementedError
