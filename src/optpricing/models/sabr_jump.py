from __future__ import annotations

from typing import Any

from optpricing.models.base import BaseModel, ParamValidator

__doc__ = """
Defines the SABR model with an added log-normal jump component.
"""


class SABRJumpModel(BaseModel):
    """SABR model with an added log-normal jump component on the spot process."""

    name: str = "SABR with Jumps"
    supports_sde: bool = True
    has_variance_process: bool = True
    has_jumps: bool = True
    is_sabr: bool = True

    default_params = {
        "alpha": 0.5,
        "beta": 0.8,
        "rho": -0.6,
        "lambda": 0.4,
        "mu_j": -0.1,
        "sigma_j": 0.15,
    }

    def __init__(self, params: dict[str, float] | None = None):
        """
        Initializes the SABR with Jumps model.

        Parameters
        ----------
        params : dict[str, float] | None, optional
            A dictionary of model parameters. If None, `default_params` are used.
            Defaults to None.
        """
        super().__init__(params or self.default_params)

    def _validate_params(self) -> None:
        p = self.params
        req = ["alpha", "beta", "rho", "lambda", "mu_j", "sigma_j"]
        ParamValidator.require(p, req, model=self.name)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SABRJumpModel):
            return NotImplemented
        return self.params == other.params

    def __hash__(self) -> int:
        return hash((self.__class__, tuple(sorted(self.params.items()))))

    #  Abstract Method Implementations
    def _sde_impl(self, **kwargs: Any) -> Any:
        raise NotImplementedError(
            "SABR uses a specialized kernel, not a generic stepper."
        )

    def _cf_impl(self, **kwargs: Any) -> Any:
        raise NotImplementedError

    def _pde_impl(self, **kwargs: Any) -> Any:
        raise NotImplementedError

    def _closed_form_impl(self, **kwargs: Any) -> Any:
        raise NotImplementedError
