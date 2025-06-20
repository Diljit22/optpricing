# src/quantfin/parity/parity_model.py

from __future__ import annotations
import math
from typing import Tuple, Any, Optional

from quantfin.models.base import BaseModel

class ParityModel(BaseModel):
    """
    A utility model providing calculations based on Put-Call Parity.

    This class is not a traditional pricing model but uses the `BaseModel`
    interface to provide parity-based calculations, such as finding a
    complementary option price.
    """
    name: str = "Put-Call Parity"
    has_closed_form: bool = True
    
    # Define the required inputs for the closed-form solver.
    cf_kwargs = ("option_price",)

    def _validate_params(self) -> None:
        """This model has no intrinsic parameters to validate."""
        pass

    def _closed_form_impl(self, *, spot: float, strike: float, r: float, t: float, call: bool, option_price: float, q: float = 0.0, **_: Any) -> float:
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
            The price of the complementary option (put if call was given, and vice-versa).
        """
        discounted_spot = spot * math.exp(-q * t)
        discounted_strike = strike * math.exp(-r * t)
        
        # Parity: C - P = S*exp(-qT) - K*exp(-rT)
        parity_difference = discounted_spot - discounted_strike

        if call:
            # We were given a Call price (C), we want the Put price (P).
            # P = C - (S*exp(-qT) - K*exp(-rT))
            return option_price - parity_difference
        else:
            # We were given a Put price (P), we want the Call price (C).
            # C = P + (S*exp(-qT) - K*exp(-rT))
            return option_price + parity_difference

    # Note: The other methods are utilities and not part of the standard
    # BaseModel pricing flow, so they are kept as regular methods.
    def price_bounds(self, *, spot: float, strike: float, r: float, t: float, call: bool, option_price: float) -> Tuple[float, float]:
        """
        Return absolute (lower, upper) no-arbitrage bounds implied by parity.
        """
        discounted_strike = strike * math.exp(-r * t)
        
        if call:
            lower_bound = max(0, spot - discounted_strike)
            upper_bound = spot
        else: # Put
            lower_bound = max(0, discounted_strike - spot)
            upper_bound = discounted_strike
            
        return lower_bound, upper_bound

    def lower_bound_rate(self, *, call_price: float, put_price: float, spot: float, strike: float, t: float) -> float:
        """
        Calculates the minimum risk-free rate `r` to avoid arbitrage.
        """
        val = strike / (spot - call_price + put_price)
        if val <= 0:
            raise ValueError("Arbitrage opportunity exists (S - C + P >= K). Cannot compute implied rate.")
        return math.log(val) / t

    # --- Abstract Method Implementations ---
    def _cf_impl(self, *args: Any, **kwargs: Any) -> Any: raise NotImplementedError
    def _sde_impl(self) -> Any: raise NotImplementedError
    def _pde_impl(self) -> Any: raise NotImplementedError