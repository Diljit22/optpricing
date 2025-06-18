from __future__ import annotations
import numpy as np
from scipy.special import gamma as gamma_func
from typing import Any, Callable

from quantfin.models.base import BaseModel, ParamValidator, CF


class CGMYModel(BaseModel):
    """
    CGMY (Carr, Geman, Madan, Yor, 2002) pure-jump Lévy model.
    """
    name = "CGMY"
    supports_cf = True
    is_pure_levy = True

    def _validate_params(self) -> None:
        p = self.params
        # C: level, G: left tail decay, M: right tail decay, Y: tail fatness
        req = ["C", "G", "M", "Y"]
        ParamValidator.require(p, req, model=self.name)
        ParamValidator.positive(p, ["C", "G", "M"], model=self.name)
        if not (p["Y"] < 2):
            raise ValueError("CGMY parameter Y must be less than 2.")
        if p["Y"] <= 0 and p["C"] != 0:
             print(f"Warning: CGMY with Y<=0 and C!=0 can have infinite moments.")

    def _cf_impl(self, *, t: float, spot: float, r: float, q: float) -> CF:
        """
        CGMY characteristic function for log-spot price log(S_t).
        """
        p = self.params
        C, G, M, Y = p["C"], p["G"], p["M"], p["Y"]

        # Risk-neutral drift compensator
        # This is phi(-i)
        compensator_val = C * gamma_func(-Y) * ((M - 1)**Y - M**Y + (G + 1)**Y - G**Y)
        drift = r - q - compensator_val

        def phi(u: np.ndarray | complex) -> np.ndarray | complex:
            term1 = 1j * u * (np.log(spot) + drift * t)
            term2 = C * t * gamma_func(-Y) * ((M - 1j * u)**Y - M**Y + (G + 1j * u)**Y - G**Y)
            return np.exp(term1 + term2)

        return phi
    
    def _sde_impl(self) -> Any:
        raise NotImplementedError(f"{self.name} SDE not implemented")

    def _pde_impl(self) -> Any:
        raise NotImplementedError(f"{self.name} PDE not implemented")

    def _closed_form_impl(self) -> Any:
        raise NotImplementedError(f"{self.name} Closed Form not implemented")