from __future__ import annotations

import math
from typing import Any

from optpricing.models.base import BaseModel, ParamValidator

__doc__ = """
Defines a model for pricing a perpetual American put option.
"""


class PerpetualPutModel(BaseModel):
    """
    Provides a closed-form price for a perpetual American put option.

    A perpetual option has no expiry date. This model assumes the holder will
    exercise optimally. The risk-free rate and volatility are considered
    intrinsic parameters of the model itself.
    """

    name: str = "Perpetual Put"
    has_closed_form: bool = True

    default_params = {"sigma": 0.20, "rate": 0.08}

    def __init__(self, params: dict[str, float]) -> None:
        """
        Initializes the model with its parameters.

        Parameters
        ----------
        params : dict[str, float] | None, optional
            A dictionary requiring 'sigma' and 'rate'. If None, `default_params`
            are used.
        """
        super().__init__(params=params)

    def _validate_params(self) -> None:
        """Validates that 'sigma' and 'rate' are present and positive."""
        ParamValidator.require(self.params, ["sigma", "rate"], model=self.name)
        ParamValidator.positive(self.params, ["sigma", "rate"], model=self.name)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PerpetualPutModel):
            return NotImplemented
        return self.params == other.params

    def __hash__(self) -> int:
        return hash((self.__class__, tuple(sorted(self.params.items()))))

    def _closed_form_impl(
        self,
        *,
        spot: float,
        strike: float,
        q: float,
        **_: Any,
    ) -> float:
        """
        Calculates the price of a perpetual American put.

        Note: This model uses the 'rate' from its own parameters (`self.params`),
        ignoring the risk-free rate passed in from the pricing technique. This
        is a specific design choice for this model. The `t` and `call`
        parameters are also ignored as they are not applicable.

        Parameters
        ----------
        spot : float
            The current price of the underlying asset.
        strike : float
            The strike price of the option.
        q : float
            The continuously compounded dividend yield of the asset.

        Returns
        -------
        float
            The price of the perpetual American put.
        """
        # uses its own intrinsic rate, not the one from the Rate object.
        r = self.params["rate"]
        sigma = self.params["sigma"]
        vol_sq = sigma**2

        # negative root of the characteristic quadratic equation
        b = r - q - 0.5 * vol_sq
        gamma = (-b - math.sqrt(b**2 + 2 * r * vol_sq)) / vol_sq

        # Calculate the optimal exercise boundary (S*)
        s_star = strike * gamma / (gamma - 1.0)

        if spot <= s_star:
            # If it is optimal to exercise, the value is the intrinsic value.
            return strike - spot
        else:
            # Otherwise, the value is given by the standard formula.
            return (strike - s_star) * (spot / s_star) ** gamma

    #  Abstract Method Implementations
    def _cf_impl(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        raise NotImplementedError(
            f"{self.name} does not support a characteristic function."
        )

    def _sde_impl(self) -> Any:
        raise NotImplementedError(f"{self.name} does not support SDE sampling.")

    def _pde_impl(self) -> Any:
        raise NotImplementedError(f"{self.name} does not support PDE solving.")
