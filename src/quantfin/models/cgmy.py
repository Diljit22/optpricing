# src/quantfin/models/cgmy.py

from __future__ import annotations
import numpy as np
from scipy.special import gamma as gamma_func
from typing import Any, Callable, Dict

from quantfin.models.base import BaseModel, ParamValidator, CF

class CGMYModel(BaseModel):
    """
    CGMY (Carr, Geman, Madan, Yor, 2002) pure-jump Lévy model.

    This is a flexible four-parameter model that can capture skewness and kurtosis.
    It has a known characteristic function for all valid parameters. Monte Carlo
    simulation is supported for the special case Y=1 (Variance Gamma process).
    """
    name: str = "CGMY"
    supports_cf: bool = True
    is_pure_levy: bool = True

    def _validate_params(self) -> None:
        """Validates the C, G, M, and Y parameters."""
        p = self.params
        req = ["C", "G", "M", "Y"]
        ParamValidator.require(p, req, model=self.name)
        ParamValidator.positive(p, ["C", "G", "M"], model=self.name)
        if not (p["Y"] < 2):
            raise ValueError("CGMY parameter Y must be less than 2 for finite variance.")
        if p["Y"] <= 0 and p["C"] != 0:
             print(f"Warning: CGMY with Y<=0 and C!=0 can have infinite moments.")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CGMYModel): return NotImplemented
        return self.params == other.params

    def __hash__(self) -> int:
        return hash((self.__class__, tuple(sorted(self.params.items()))))

    def _cf_impl(self, *, t: float, spot: float, r: float, q: float, **_: Any) -> CF:
        """Risk-neutral characteristic function for the log-spot price log(S_t)."""
        compensator = np.log(self.raw_cf(t=1.0)(-1j))
        drift = r - q - np.real(compensator)
        
        def phi(u: np.ndarray | complex) -> np.ndarray | complex:
            return self.raw_cf(t=t)(u) * np.exp(1j * u * (np.log(spot) + drift * t))
        return phi

    def raw_cf(self, *, t: float) -> Callable:
        """Returns the CF of the raw CGMY process, without drift or spot."""
        p = self.params
        C, G, M, Y = p["C"], p["G"], p["M"], p["Y"]
        def phi_raw(u: np.ndarray | complex) -> np.ndarray | complex:
            term = C * t * gamma_func(-Y) * ((M - 1j * u)**Y - M**Y + (G + 1j * u)**Y - G**Y)
            return np.exp(term)
        return phi_raw

    def sample_terminal_log_return(self, T: float, size: int, rng: np.random.Generator) -> np.ndarray:
        """
        Draws samples from the terminal distribution of the raw CGMY process.
        
        NOTE: This is only implemented for the special case Y=1, where the
        process is a difference of two time-changed Brownian motions.
        """
        p = self.params
        C, G, M, Y = p["C"], p["G"], p["M"], p["Y"]

        if not np.isclose(Y, 1.0):
            raise NotImplementedError("Monte Carlo sampling for CGMY is only implemented for Y=1.")

        # For Y=1, CGMY is a scaled difference of two Gamma processes.
        # This is equivalent to a Variance Gamma process.
        # We can sample this by time-changing a Brownian motion.
        g_up = rng.gamma(shape=C*T, scale=1/M, size=size)
        g_down = rng.gamma(shape=C*T, scale=1/G, size=size)
        
        return g_up - g_down

    # --- Abstract Method Implementations ---
    def _sde_impl(self, **kwargs: Any) -> Any: raise NotImplementedError("CGMY is a pure Lévy process, use terminal sampling.")
    def _pde_impl(self, **kwargs: Any) -> Any: raise NotImplementedError
    def _closed_form_impl(self, **kwargs: Any) -> Any: raise NotImplementedError