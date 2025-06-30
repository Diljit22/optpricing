from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.stats import norm

from optpricing.models.base import CF, BaseModel, ParamValidator, PDECoeffs

__doc__ = """
Defines the Black-Scholes-Merton (BSM) model for pricing European options.
"""


class BSMModel(BaseModel):
    """
    Black-Scholes-Merton (BSM) model for pricing European options.

    This model assumes the underlying asset follows a geometric Brownian motion
    with constant volatility and risk-free rate.
    """

    name: str = "Black-Scholes-Merton"
    supports_cf: bool = True
    supports_sde: bool = True
    supports_pde: bool = True
    has_closed_form: bool = True

    default_params = {"sigma": 0.2}
    param_defs = {
        "sigma": {
            "label": "Volatility",
            "default": 0.2,
            "min": 0.01,
            "max": 2.0,
            "step": 0.01,
        }
    }

    def __init__(self, params: dict[str, float] | None = None):
        """
        Initializes the BSM model.

        Parameters
        ----------
        params : dict[str, float] | None, optional
            A dictionary of model parameters. If None, `default_params` are used.
            Must contain 'sigma'.
        """
        super().__init__(params or self.default_params)

    def _validate_params(self) -> None:
        """Validate the 'sigma' parameter."""
        ParamValidator.require(self.params, ["sigma"], model=self.name)
        ParamValidator.positive(self.params, ["sigma"], model=self.name)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BSMModel):
            return NotImplemented
        return self.params == other.params

    def __hash__(self) -> int:
        return hash((self.__class__, tuple(sorted(self.params.items()))))

    def _closed_form_impl(
        self,
        *,
        spot: float,
        strike: float,
        r: float,
        q: float,
        t: float,
        call: bool = True,
    ) -> float:
        """
        Computes the Black-Scholes-Merton price in closed form.

        Parameters
        ----------
        spot : float
            The current price of the underlying asset.
        strike : float
            The strike price of the option.
        r : float
            The continuously compounded risk-free rate.
        q : float
            The continuously compounded dividend yield.
        t : float
            The time to maturity of the option, in years.
        call : bool, optional
            True for a call option, False for a put. Defaults to True.

        Returns
        -------
        float
            The price of the European option.
        """
        sigma = self.params["sigma"]
        sqrt_t = np.sqrt(t)
        d1 = (np.log(spot / strike) + (r - q + 0.5 * sigma**2) * t) / (sigma * sqrt_t)
        d2 = d1 - sigma * sqrt_t

        df_div = np.exp(-q * t)
        df_rate = np.exp(-r * t)

        if call:
            price = spot * df_div * norm.cdf(d1) - strike * df_rate * norm.cdf(d2)
        else:
            price = strike * df_rate * norm.cdf(-d2) - spot * df_div * norm.cdf(-d1)
        return price

    def delta_analytic(
        self,
        *,
        spot: float,
        strike: float,
        r: float,
        q: float,
        t: float,
        call: bool = True,
    ) -> float:
        """Analytic delta for the BSM model."""
        sigma = self.params["sigma"]
        d1 = (np.log(spot / strike) + (r - q + 0.5 * sigma**2) * t) / (
            sigma * np.sqrt(t)
        )
        df_div = np.exp(-q * t)
        return df_div * norm.cdf(d1) if call else -df_div * norm.cdf(-d1)

    def gamma_analytic(
        self,
        *,
        spot: float,
        strike: float,
        r: float,
        q: float,
        t: float,
    ) -> float:
        """Analytic gamma for the BSM model."""
        sigma = self.params["sigma"]
        sqrt_t = np.sqrt(t)
        d1 = (np.log(spot / strike) + (r - q + 0.5 * sigma**2) * t) / (sigma * sqrt_t)
        df_div = np.exp(-q * t)
        return df_div * norm.pdf(d1) / (spot * sigma * sqrt_t)

    def vega_analytic(
        self,
        *,
        spot: float,
        strike: float,
        r: float,
        q: float,
        t: float,
    ) -> float:
        """Analytic vega for the BSM model."""
        sigma = self.params["sigma"]
        sqrt_t = np.sqrt(t)
        d1 = (np.log(spot / strike) + (r - q + 0.5 * sigma**2) * t) / (sigma * sqrt_t)
        return spot * np.exp(-q * t) * norm.pdf(d1) * sqrt_t

    def theta_analytic(
        self,
        *,
        spot: float,
        strike: float,
        r: float,
        q: float,
        t: float,
        call: bool = True,
    ) -> float:
        """Analytic theta for the BSM model."""
        sigma = self.params["sigma"]
        sqrt_t = np.sqrt(t)
        d1 = (np.log(spot / strike) + (r - q + 0.5 * sigma**2) * t) / (sigma * sqrt_t)
        d2 = d1 - sigma * sqrt_t
        term1 = -spot * np.exp(-q * t) * norm.pdf(d1) * sigma / (2 * sqrt_t)
        if call:
            term2 = q * spot * np.exp(-q * t) * norm.cdf(d1)
            term3 = -r * strike * np.exp(-r * t) * norm.cdf(d2)
        else:
            term2 = -q * spot * np.exp(-q * t) * norm.cdf(-d1)
            term3 = r * strike * np.exp(-r * t) * norm.cdf(-d2)
        return term1 + term2 + term3

    def rho_analytic(
        self,
        *,
        spot: float,
        strike: float,
        r: float,
        q: float,
        t: float,
        call: bool = True,
    ) -> float:
        """Analytic rho for the BSM model."""
        sigma = self.params["sigma"]
        sqrt_t = np.sqrt(t)
        d2 = (np.log(spot / strike) + (r - q - 0.5 * sigma**2) * t) / (sigma * sqrt_t)
        df_rate = np.exp(-r * t)
        return strike * t * df_rate * (norm.cdf(d2) if call else -norm.cdf(-d2))

    def _cf_impl(
        self,
        *,
        t: float,
        spot: float,
        r: float,
        q: float,
        **_,
    ) -> CF:
        """
        Returns the characteristic function phi(u) for the log-spot price log(S_t).

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
        sigma = self.params["sigma"]
        drift = r - q - 0.5 * sigma**2

        def phi(u: np.ndarray | complex) -> np.ndarray | complex:
            mean_component = 1j * u * (np.log(spot) + drift * t)
            variance_component = -0.5 * (u**2) * (sigma**2) * t
            return np.exp(mean_component + variance_component)

        return phi

    def _sde_impl(self) -> Callable:
        """Returns the Euler-Maruyama stepper for the BSM log-price process."""
        sigma = self.params["sigma"]

        def stepper(
            log_s_t: np.ndarray, r: float, q: float, dt: float, dw_t: np.ndarray
        ) -> np.ndarray:
            drift = (r - q - 0.5 * sigma**2) * dt
            diffusion = sigma * dw_t
            return log_s_t + drift + diffusion

        return stepper

    def _pde_impl(self) -> PDECoeffs:
        """Returns the Black-Scholes PDE coefficients."""
        sigma = self.params["sigma"]

        def coeffs(
            S: np.ndarray, r: float, q: float
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            return sigma**2 * S**2, (r - q) * S, -r * np.ones_like(S)

        return coeffs
