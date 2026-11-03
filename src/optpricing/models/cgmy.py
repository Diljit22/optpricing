from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from scipy.special import gamma as gamma_func

from optpricing.models.base import CF, BaseModel, ParamValidator, sample_from_cf

__doc__ = """
Defines the CGMY (Carr, Geman, Madan, Yor, 2002) pure-jump Levy model.
"""


class CGMYModel(BaseModel):
    """
    CGMY (Carr, Geman, Madan, Yor, 2002) pure-jump Levy model.

    This is a flexible four-parameter model that can capture skewness and kurtosis.
    It has a known characteristic function for all valid parameters (``Y < 2``),
    including the limiting cases ``Y = 0`` (difference of two gamma processes,
    i.e. Variance Gamma) and ``Y = 1``. Monte Carlo sampling is exact for ``Y = 0``
    and uses inverse-transform sampling of the terminal characteristic function
    otherwise.
    """

    name: str = "CGMY"
    supports_cf: bool = True
    is_pure_levy: bool = True

    default_params = {"C": 0.02, "G": 5.0, "M": 5.0, "Y": 1.2}

    def __init__(self, params: dict[str, float] | None = None):
        """
        Initializes the CGMY model.

        Parameters
        ----------
        params : dict[str, float] | None, optional
            A dictionary of model parameters. If None, `default_params` are used.
            Defaults to None.
        """
        super().__init__(params or self.default_params)

    def _validate_params(self) -> None:
        """Validates the C, G, M, and Y parameters."""
        p = self.params
        req = ["C", "G", "M", "Y"]
        ParamValidator.require(p, req, model=self.name)
        ParamValidator.positive(p, ["C", "G", "M"], model=self.name)
        if not (p["Y"] < 2):
            raise ValueError(
                "CGMY parameter Y must be less than 2 for finite variance."
            )
        if p["Y"] < 0:
            raise ValueError("CGMY parameter Y must be non-negative.")
        if not (p["M"] > 1):
            # E[exp(X_t)] = exp(t * psi(-i)) requires the right tail to decay
            # faster than exp(-x): M > 1 (Carr, Geman, Madan & Yor 2002).
            raise ValueError(
                "CGMY parameter M must exceed 1 for the risk-neutral measure to exist."
            )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CGMYModel):
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
        compensator = np.log(self.raw_cf(t=1.0)(-1j))
        drift = r - q - np.real(compensator)

        def phi(u: np.ndarray | complex) -> np.ndarray | complex:
            return self.raw_cf(t=t)(u) * np.exp(1j * u * (np.log(spot) + drift * t))

        return phi

    def raw_cf(self, *, t: float) -> Callable:
        """
        Returns the CF of the raw CGMY process, without drift or spot.

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
        C, G, M, Y = p["C"], p["G"], p["M"], p["Y"]

        # Levy exponent psi(u) with Gamma(-Y) poles at Y = 0 and Y = 1 replaced
        # by their limits: Y -> 0 gives the Variance-Gamma (gamma difference)
        # exponent, Y -> 1 gives the x*log(x) form.
        if np.isclose(Y, 0.0):

            def psi(u):
                return C * (np.log(M / (M - 1j * u)) + np.log(G / (G + 1j * u)))

        elif np.isclose(Y, 1.0):

            def psi(u):
                return C * (
                    (M - 1j * u) * np.log(M - 1j * u)
                    - M * np.log(M)
                    + (G + 1j * u) * np.log(G + 1j * u)
                    - G * np.log(G)
                )

        else:
            gamma_minus_y = gamma_func(-Y)

            def psi(u):
                return (
                    C
                    * gamma_minus_y
                    * ((M - 1j * u) ** Y - M**Y + (G + 1j * u) ** Y - G**Y)
                )

        def phi_raw(u: np.ndarray | complex) -> np.ndarray | complex:
            return np.exp(t * psi(u))

        return phi_raw

    def sample_terminal_log_return(
        self,
        T: float,
        size: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Generates samples of the terminal log-return of the raw CGMY process.

        For ``Y = 0`` the process is the difference of two independent gamma
        processes with shape ``C*T`` and rates ``M`` and ``G`` (exact). For any
        other ``Y`` the terminal law is sampled by inverse transform of the
        characteristic function (COS expansion), see
        :func:`optpricing.models.base.sample_from_cf`.

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
        C, G, M, Y = p["C"], p["G"], p["M"], p["Y"]

        if np.isclose(Y, 0.0):
            g_up = rng.gamma(shape=C * T, scale=1 / M, size=size)
            g_down = rng.gamma(shape=C * T, scale=1 / G, size=size)
            return g_up - g_down

        return sample_from_cf(self.raw_cf(t=T), size, rng)

    #  Abstract Method Implementations
    def _sde_impl(self, **kwargs: Any) -> Any:
        raise NotImplementedError("CGMY is a pure Levy process, use terminal sampling.")

    def _pde_impl(self, **kwargs: Any) -> Any:
        raise NotImplementedError

    def _closed_form_impl(self, **kwargs: Any) -> Any:
        raise NotImplementedError
