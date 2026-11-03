from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np
from scipy.special import ndtr

from optpricing.models.base import BaseModel, ParamValidator

__doc__ = """
Defines the Stochastic Alpha, Beta, Rho (SABR) model.
"""


def hagan_lognormal_vol(
    forward: float | np.ndarray,
    strike: float | np.ndarray,
    t: float,
    *,
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
) -> np.ndarray | float:
    """
    Hagan et al. (2002, eq. 2.17) lognormal implied volatility of the SABR model.

    Parameters
    ----------
    forward, strike : float | np.ndarray
        Forward and strike (both strictly positive; apply the shift beforehand
        for shifted SABR).
    t : float
        Time to expiry in years.
    alpha, beta, rho, nu : float
        SABR parameters: initial volatility, CEV exponent, correlation and
        volatility of volatility.

    Returns
    -------
    np.ndarray | float
        Black implied volatility. The expansion is asymptotic in ``t`` and in
        ``log(F/K)``; it is known to produce arbitrageable (negative-density)
        smiles for very low strikes and long maturities, which this function
        does not detect.
    """
    F = np.asarray(forward, dtype=float)
    K = np.asarray(strike, dtype=float)
    one_minus_beta = 1.0 - beta
    log_fk = np.log(F / K)
    fk = F * K
    fk_pow = fk ** (0.5 * one_minus_beta)  # (FK)^((1-beta)/2)

    correction = 1.0 + t * (
        one_minus_beta**2 / 24.0 * alpha**2 / fk**one_minus_beta
        + 0.25 * rho * beta * nu * alpha / fk_pow
        + (2.0 - 3.0 * rho**2) / 24.0 * nu**2
    )
    z = (nu / alpha) * fk_pow * log_fk
    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.log((np.sqrt(1.0 - 2.0 * rho * z + z * z) + z - rho) / (1.0 - rho))
        ratio = np.where(np.abs(z) < 1e-7, 1.0 - 0.5 * rho * z, z / x)
    denominator = fk_pow * (
        1.0
        + one_minus_beta**2 / 24.0 * log_fk**2
        + one_minus_beta**4 / 1920.0 * log_fk**4
    )
    vol = alpha / denominator * ratio * correction
    return float(vol) if vol.ndim == 0 else vol


class SABRModel(BaseModel):
    """
    SABR model of Hagan, Kumar, Lesniewski & Woodward (2002).

    The forward ``F`` and its volatility follow

        dF = sigma_t F^beta dW,   d sigma_t = nu sigma_t dZ,   d<W, Z> = rho dt,

    with ``sigma_0 = alpha``. Spot inputs are converted to the forward
    ``F = S exp((r - q) T)``; an optional ``shift`` prices ``F + shift`` (shifted
    SABR, for negative or near-zero forwards).

    Pricing: the closed form is Black-76 with Hagan's lognormal implied
    volatility (an asymptotic expansion, accurate for moderate ``nu^2 T`` and
    strikes not too far from the forward); Monte Carlo simulates the forward
    with an Euler step for ``F`` and the exact lognormal step for ``sigma_t``.

    Notes
    -----
    Before v2.5 the model had three parameters and used ``alpha`` as the
    volatility of volatility (the initial volatility defaulted to the same
    number). Such parameter sets are still accepted with a
    ``DeprecationWarning`` and mapped to ``nu = alpha``.
    """

    name: str = "SABR"
    supports_sde: bool = True
    has_variance_process: bool = True
    has_closed_form: bool = True
    is_sabr: bool = True
    default_params = {"alpha": 0.5, "beta": 0.8, "rho": -0.6, "nu": 0.5}
    param_defs = {
        "alpha": {
            "label": "Initial Vol (alpha)",
            "default": 0.5,
            "min": 0.001,
            "max": 5.0,
            "step": 0.01,
        },
        "beta": {"label": "Beta", "default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05},
        "rho": {
            "label": "Correlation",
            "default": -0.6,
            "min": -0.99,
            "max": 0.99,
            "step": 0.05,
        },
        "nu": {
            "label": "Vol of Vol (nu)",
            "default": 0.5,
            "min": 0.01,
            "max": 3.0,
            "step": 0.05,
        },
    }

    def __init__(self, params: dict[str, float] | None = None):
        """
        Initializes the SABR model.

        Parameters
        ----------
        params : dict[str, float] | None, optional
            A dictionary of model parameters. If None, `default_params` are used.
            Keys:

            - ``alpha``: initial volatility ``sigma_0``.
            - ``beta``: CEV exponent in ``[0, 1]`` (0 = normal, 1 = lognormal).
            - ``rho``: correlation between forward and volatility in ``(-1, 1)``.
            - ``nu``: volatility of volatility.
            - ``shift`` (optional): displacement for shifted SABR (default 0).
        """
        params = dict(params) if params is not None else dict(self.default_params)
        if "nu" not in params and "alpha" in params:
            warnings.warn(
                "SABRModel without 'nu' is deprecated: 'alpha' used to be the "
                "volatility of volatility. Interpreting nu=alpha; pass 'alpha' "
                "(initial volatility) and 'nu' (vol of vol) explicitly.",
                DeprecationWarning,
                stacklevel=2,
            )
            params["nu"] = params["alpha"]
        super().__init__(params)

    def _validate_params(self) -> None:
        """Validates alpha, beta, rho, nu (and the optional shift)."""
        p = self.params
        req = ["alpha", "beta", "rho", "nu"]
        ParamValidator.require(p, req, model=self.name)
        ParamValidator.positive(p, ["alpha", "nu"], model=self.name)
        ParamValidator.bounded(p, "beta", 0.0, 1.0, model=self.name)
        if not (-1.0 < p["rho"] < 1.0):
            raise ValueError(f"{self.name}: parameter 'rho' must be in (-1, 1).")
        if p.get("shift", 0.0) < 0.0:
            raise ValueError(f"{self.name}: parameter 'shift' must be non-negative.")

    @property
    def shift(self) -> float:
        return float(self.params.get("shift", 0.0))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SABRModel):
            return NotImplemented
        return self.params == other.params

    def __hash__(self) -> int:
        return hash((self.__class__, tuple(sorted(self.params.items()))))

    def implied_volatility(
        self,
        forward: float | np.ndarray,
        strike: float | np.ndarray,
        t: float,
    ) -> float | np.ndarray:
        """
        Hagan lognormal implied volatility for (shifted) forward and strike(s).

        Parameters
        ----------
        forward : float | np.ndarray
            Forward price(s) (unshifted).
        strike : float | np.ndarray
            Strike(s) (unshifted).
        t : float
            Time to expiry in years.

        Returns
        -------
        float | np.ndarray
            Black implied volatility of the shifted forward ``F + shift``.

        Raises
        ------
        ValueError
            If the (shifted) forward or strike is not positive, or the expansion
            yields a non-positive / non-finite volatility.
        """
        p = self.params
        f_shifted = np.asarray(forward, dtype=float) + self.shift
        k_shifted = np.asarray(strike, dtype=float) + self.shift
        if np.any(f_shifted <= 0.0) or np.any(k_shifted <= 0.0):
            raise ValueError(
                f"{self.name}: shifted forward and strike must be positive "
                "(increase 'shift' for negative or zero forwards)."
            )
        vol = hagan_lognormal_vol(
            f_shifted,
            k_shifted,
            t,
            alpha=p["alpha"],
            beta=p["beta"],
            rho=p["rho"],
            nu=p["nu"],
        )
        if not np.all(np.isfinite(vol)) or np.any(np.asarray(vol) <= 0.0):
            raise ValueError(
                f"{self.name}: Hagan's expansion yields a non-positive or undefined "
                "implied volatility for these inputs; use Monte Carlo instead."
            )
        return vol

    def _closed_form_impl(
        self,
        *,
        spot: float,
        strike: float,
        r: float,
        q: float,
        t: float,
        call: bool = True,
        **_: Any,
    ) -> float:
        """
        Black-76 price with Hagan's implied volatility.

        Parameters
        ----------
        spot : float
            Current spot; converted to the forward ``S exp((r-q) t)``.
        strike : float
            Strike price.
        r : float
            Continuously compounded risk-free rate.
        q : float
            Continuously compounded dividend yield.
        t : float
            Time to maturity in years.
        call : bool, optional
            True for a call, False for a put. Defaults to True.

        Returns
        -------
        float
            The European option price.
        """
        forward = spot * math.exp((r - q) * t)
        sigma = float(self.implied_volatility(forward, strike, t))
        f_s, k_s = forward + self.shift, strike + self.shift
        sqrt_t = math.sqrt(t)
        d1 = (math.log(f_s / k_s) + 0.5 * sigma**2 * t) / (sigma * sqrt_t)
        d2 = d1 - sigma * sqrt_t
        discount = math.exp(-r * t)
        if call:
            return discount * (f_s * ndtr(d1) - k_s * ndtr(d2))
        return discount * (k_s * ndtr(-d2) - f_s * ndtr(-d1))

    #  Abstract Method Implementations
    def _sde_impl(self, **kwargs: Any) -> Any:
        raise NotImplementedError(
            "SABR uses a specialized kernel, not a generic stepper."
        )

    def _cf_impl(self, **kwargs: Any) -> Any:
        raise NotImplementedError

    def _pde_impl(self, **kwargs: Any) -> Any:
        raise NotImplementedError
