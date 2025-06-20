# src/quantfin/models/sabr_jump.py

from __future__ import annotations
from typing import Any, Callable, Dict
import numpy as np

from quantfin.models.base import BaseModel, ParamValidator

class SABRJumpModel(BaseModel):
    """SABR model with an added log-normal jump component on the spot process."""
    name: str = "SABR with Jumps"
    supports_sde: bool = True
    has_variance_process: bool = True
    has_jumps: bool = True
    is_sabr: bool = True # Special flag for the MC dispatcher

    def _validate_params(self) -> None:
        p = self.params
        req = ["alpha", "beta", "rho", "lambda", "mu_j", "sigma_j"]
        ParamValidator.require(p, req, model=self.name)
        # Add other validations as needed...

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SABRJumpModel): return NotImplemented
        return self.params == other.params

    def __hash__(self) -> int:
        return hash((self.__class__, tuple(sorted(self.params.items()))))

    # --- Abstract Method Implementations ---
    def _sde_impl(self, **kwargs: Any) -> Any: raise NotImplementedError("SABR uses a specialized kernel, not a generic stepper.")
    def _cf_impl(self, **kwargs: Any) -> Any: raise NotImplementedError
    def _pde_impl(self, **kwargs: Any) -> Any: raise NotImplementedError
    def _closed_form_impl(self, **kwargs: Any) -> Any: raise NotImplementedError