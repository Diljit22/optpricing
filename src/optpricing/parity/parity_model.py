from __future__ import annotations

import math
from typing import Any

from optpricing.models.base import BaseModel

__doc__ = """
This module provides a utility model for calculations based on Put-Call Parity.
"""


class ParityModel(BaseModel):
    """
    A utility model providing calculations based on Put-Call Parity.

    This class is not a traditional pricing model but uses the `BaseModel`
    interface to provide parity-based calculations, such as finding a
    complementary option price.
    """

    name: str = "Put-Call Parity"
    has_closed_form: bool = True

    # Define inputs for the closed-form solver
    cf_kwargs = ("option_price",)

    def _validate_params(self) -> None:
        """This model has no intrinsic parameters to validate."""
        pass

    def _closed_form_impl(
        self,
        *,
        spot: float,
        strike: float,
        r: float,
        t: float,
        call: bool,
        option_price: float,
        q: float = 0.0,
        **_: Any,
    ) -> float:
        """
        Return the complementary price implied by put-call parity.

        Parameters
        ----------
        spot : float
            The current price of the underlying asset.
        strike : float
            The strike price of the option.
        r : float
            The risk-free rate.
        t : float
            The time to maturity.
        call : bool
            True if `option_price` is for a call, False if it's for a put.
        option_price : float
            The price of the known option.
        q : float, optional
            The continuously compounded dividend yield, by default 0.0.

        Returns
        -------
        float
            The price of the complementary option (i.e. put -> call).
        """
        discounted_spot = spot * math.exp(-q * t)
        discounted_strike = strike * math.exp(-r * t)

        # Parity: C - P = S*exp(-qT) - K*exp(-rT)
        parity_difference = discounted_spot - discounted_strike

        if call:
            return option_price - parity_difference
        else:
            return option_price + parity_difference

    def price_bounds(
        self,
        *,
        spot: float,
        strike: float,
        r: float,
        t: float,
        call: bool,
        option_price: float,  # noqa: F841
    ) -> tuple[float, float]:
        """
        Return absolute (lower, upper) no-arbitrage bounds for an option.

        Parameters
        ----------
        spot : float
            The current price of the underlying asset.
        strike : float
            The strike price of the option.
        r : float
            The risk-free rate.
        t : float
            The time to maturity.
        call : bool
            True if `option_price` is for a call, False if it's for a put.
        option_price : float
            The price of the known option. In this case it is a place holder.

        Returns
        -------
        tuple[float, float]
            A tuple containing the (lower_bound, upper_bound) for the option price.
        """
        discounted_strike = strike * math.exp(-r * t)

        if call:
            lower_bound = max(0, spot - discounted_strike)
            upper_bound = spot
        else:  # Put
            lower_bound = max(0, discounted_strike - spot)
            upper_bound = discounted_strike

        return lower_bound, upper_bound

    def lower_bound_rate(
        self,
        *,
        call_price: float,
        put_price: float,
        spot: float,
        strike: float,
        t: float,
    ) -> float:
        """
        Calculates the minimum risk-free rate `r` to avoid arbitrage.

        Parameters
        ----------
        call_price : float
            The price of the call option.
        put_price : float
            The price of the put option.
        spot : float
            The current price of the underlying asset.
        strike : float
            The strike price of the option.
        t : float
            The time to maturity.

        Returns
        -------
        float
            The minimum continuously compounded risk-free rate to avoid arbitrage.

        Raises
        ------
        ValueError
            If an arbitrage opportunity already exists (S - C + P >= K).
        """
        val = strike / (spot - call_price + put_price)
        if val <= 0:
            raise ValueError(
                "Arbitrage exists (S - C + P >= K). Cannot compute implied rate."
            )
        return math.log(val) / t

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
