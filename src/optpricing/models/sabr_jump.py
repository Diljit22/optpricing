from __future__ import annotations

import warnings
from typing import Any

from optpricing.models.base import BaseModel, ParamValidator

__doc__ = """
Defines the SABR model with an added log-normal jump component.
"""


class SABRJumpModel(BaseModel):
    """
    SABR forward dynamics (see :class:`~optpricing.models.SABRModel`) with
    compensated log-normal jumps on the forward:

        dF/F- = -lambda k dt + sigma_t F^(beta-1) dW + (e^J - 1) dN,
        J ~ N(mu_j, sigma_j^2),  k = E[e^J] - 1.

    Monte Carlo only; the three-parameter legacy form (``alpha`` as vol of vol)
    is mapped to ``nu = alpha`` with a ``DeprecationWarning``.
    """

    name: str = "SABR with Jumps"
    supports_sde: bool = True
    has_variance_process: bool = True
    has_jumps: bool = True
    is_sabr: bool = True

    default_params = {
        "alpha": 0.5,
        "beta": 0.8,
        "rho": -0.6,
        "nu": 0.5,
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
        params = dict(params) if params is not None else dict(self.default_params)
        if "nu" not in params and "alpha" in params:
            warnings.warn(
                "SABRJumpModel without 'nu' is deprecated: 'alpha' used to be the "
                "volatility of volatility. Interpreting nu=alpha; pass 'alpha' "
                "(initial volatility) and 'nu' (vol of vol) explicitly.",
                DeprecationWarning,
                stacklevel=2,
            )
            params["nu"] = params["alpha"]
        super().__init__(params)

    def _validate_params(self) -> None:
        p = self.params
        req = ["alpha", "beta", "rho", "nu", "lambda", "mu_j", "sigma_j"]
        ParamValidator.require(p, req, model=self.name)
        ParamValidator.positive(p, ["alpha", "nu", "sigma_j"], model=self.name)
        ParamValidator.bounded(p, "beta", 0.0, 1.0, model=self.name)
        if not (-1.0 < p["rho"] < 1.0):
            raise ValueError(f"{self.name}: parameter 'rho' must be in (-1, 1).")
        if p["lambda"] < 0.0:
            raise ValueError(f"{self.name}: parameter 'lambda' must be non-negative.")
        if p.get("shift", 0.0) < 0.0:
            raise ValueError(f"{self.name}: parameter 'shift' must be non-negative.")

    @property
    def shift(self) -> float:
        return float(self.params.get("shift", 0.0))

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
