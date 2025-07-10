from __future__ import annotations

import math
from typing import Any

from scipy.optimize import brentq

from optpricing.models.base import BaseModel

__doc__ = """
This module defines a model to find the risk-free rate implied by put-call parity.
"""


class ImpliedRateModel(BaseModel):
    """
    Calculates the risk-free rate implied by put-call parity for European options.

    This model solves for the risk-free rate `r` that satisfies the equation:
    C - P = S*exp(-qT) - K*exp(-rT)
    """

    name: str = "Implied Rate"
    has_closed_form: bool = True

    # Define inputs for the closed-form solver
    cf_kwargs = ("call_price", "put_price")

    def _validate_params(self) -> None:
        """This model has no intrinsic parameters to validate."""
        pass

    def _closed_form_impl(
        self,
        *,
        call_price: float,
        put_price: float,
        spot: float,
        strike: float,
        t: float,
        q: float = 0.0,
        **_: Any,
    ) -> float:
        """
        Solves for the implied risk-free rate using a root-finding algorithm.

        Parameters
        ----------
        call_price : float
            The market price of the call option.
        put_price : float
            The market price of the put option.
        spot : float
            The current price of the underlying asset.
        strike : float
            The strike price of the options.
        t : float
            The time to maturity of the options.
        q : float, optional
            The continuously compounded dividend yield, by default 0.0.

        Returns
        -------
        float
            The implied continuously compounded risk-free rate.

        Raises
        ------
        ValueError
            If a root cannot be bracketed within a reasonable range.
        """
        discounted_spot = spot * math.exp(-q * t)
        price_difference = call_price - put_price

        def objective_func(
            r: float,
        ) -> float:
            """The put-call parity equation rearranged to equal zero."""
            return discounted_spot - strike * math.exp(-r * t) - price_difference

        # Attempt to find a bracket for the root
        low, high = -0.5, 0.5
        f_low, f_high = objective_func(low), objective_func(high)

        for _ in range(10):  # Try up to 10 times to expand the bracket
            if f_low * f_high < 0:
                break
            low -= 0.5
            high += 0.5
            f_low, f_high = objective_func(low), objective_func(high)
        else:
            raise ValueError(
                "Unable to bracket a root for implied rate. Check input for arbitrage."
            )

        return brentq(objective_func, low, high, xtol=1e-9, maxiter=100)

    # Abstract Method Implementations
    def _cf_impl(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        raise NotImplementedError

    def _sde_impl(self) -> Any:
        raise NotImplementedError

    def _pde_impl(self) -> Any:
        raise NotImplementedError
