from __future__ import annotations

import math
from typing import Tuple, Any, Optional

from quantfin.models.base import BaseModel



class ParityModel(BaseModel):
    """Put-Call parity utilities for European (and dividend-less American) options.

    Functions provided
    ------------------
    * Complementary price via parity
    * No-arbitrage price bounds
    * Lower bound on the risk-free rate
    """

    name = "Put-Call Parity"
    supports_cf = False
    supports_sde = False
    supports_pde = False
    has_closed_form = True
    supported_lattices: set[str] = set()
    cf_kwargs = BaseModel.cf_kwargs + ("option_price", "complementary_price")


    def _validate_params(self) -> None:  # ParityModel has no tunable params
        pass


    def _closed_form_impl(
        self,
        *,
        spot: float,
        strike: float,
        r: float,
        t: float,
        call: bool = True,
        option_price: Optional[float] = None,
        complementary_price: Optional[float] = None,
        q: float = 0.0,
    ) -> float:
        """Return the complementary price implied by put-call parity.

        Exactly **one** of ``option_price`` *or* ``complementary_price`` must
        be supplied.
        """
        if (option_price is None) == (complementary_price is None):
            raise ValueError("Provide exactly one of option_price or complementary_price.")

        discK = strike * math.exp(-r * t)
        discS = spot * math.exp(-q * t)
        diff = discS - discK  # S*e^{-qT} - K*e^{-rT}

        if option_price is not None:
            # we supplied the price of *call* if call=True else *put*
            return (option_price - diff) if call else (option_price + diff)
        else:
            # we supplied the complementary price
            return (complementary_price + diff) if call else (complementary_price - diff)


    def price_bounds(
        self,
        *,
        spot: float,
        strike: float,
        r: float,
        t: float,
        call: bool,
        option_price: float,
    ) -> Tuple[float, float]:
        """Return absolute (lower, upper) no-arbitrage bounds implied by parity."""
        adjK = strike * math.exp(-r * t)
        mx = spot - adjK
        mn = spot - strike

        lower = (mn + option_price) if call else (option_price - mx)
        upper = (mx + option_price) if call else (option_price - mn)
        return max(lower, 0.0), upper

    def lower_bound_rate(
        self,
        *,
        call_price: float,
        put_price: float,
        spot: float,
        strike: float,
        t: float,
    ) -> float:
        """Minimum *r* that keeps the parity inequality non-negative."""
        val = spot - call_price + put_price
        if val <= 0.0:
            raise ValueError("S - C + P must be > 0 to compute implied rate.")
        return -math.log(val / strike) / t
    
    
    def _cf_impl(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(f"{self.name} characteristic function not implemented")

    def _sde_impl(self) -> Any:
        raise NotImplementedError(f"{self.name} SDE not implemented")

    def _pde_impl(self) -> Any:
        raise NotImplementedError(f"{self.name} PDE not implemented")