from __future__ import annotations

from typing import Any

from optpricing.models.base import BaseModel, ParamValidator

__doc__ = """
Defines the Stochastic Alpha, Beta, Rho (SABR) model.
"""


class SABRModel(BaseModel):
    """
    Stochastic Alpha, Beta, Rho (SABR) model from Hagan et al. (2002).
    """

    name: str = "SABR"
    supports_sde: bool = True
    has_variance_process: bool = True
    is_sabr: bool = True
    default_params = {"alpha": 0.5, "beta": 0.8, "rho": -0.6}

    def __init__(self, params: dict[str, float] | None = None):
        """
        Initializes the SABR model.

        Parameters
        ----------
        params : dict[str, float] | None, optional
            A dictionary of model parameters. If None, `default_params` are used.
        """
        super().__init__(params or self.default_params)

    def _validate_params(self) -> None:
        """Validates that 'alpha', 'beta', and 'rho' are present and positive."""
        p = self.params
        req = ["alpha", "beta", "rho"]
        ParamValidator.require(p, req, model=self.name)
        ParamValidator.positive(p, ["alpha"], model=self.name)
        ParamValidator.bounded(p, "beta", 0.0, 1.0, model=self.name)
        ParamValidator.bounded(p, "rho", -1.0, 1.0, model=self.name)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SABRModel):
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
