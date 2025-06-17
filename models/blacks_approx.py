from __future__ import annotations
import math
import numpy as np
from typing import Optional, Any
from scipy.stats import norm  # Use norm for consistency with BSMModel

from models.base.base_model import BaseModel
from models.base.validators import ParamValidator

class BlacksApproxModel(BaseModel):
    """
    Black's (1975) approximation for an American call option on a stock paying discrete dividends.
    The price is the maximum of:
    1. European call on a stock with price reduced by PV of all dividends.
    2. European calls expiring just before each dividend payment, on a stock with price reduced
       by PV of dividends up to that point.
    """
    name = "Black's Approximation"
    supports_cf     = False
    supports_sde    = False
    supports_pde    = False
    has_closed_form = True
    supported_lattices: set[str] = set()
    cf_kwargs = BaseModel.cf_kwargs + ("discrete_dividends", "ex_div_times")

    def _validate_params(self) -> None:
        p = self.params
        ParamValidator.require(p, ["sigma"], model=self.name)
        ParamValidator.positive(p, ["sigma"], model=self.name)

    def _closed_form_impl(
        self,
        *,
        spot: float,
        strike: float,
        r: float,
        t: float,
        call: bool = True,
        discrete_dividends: np.ndarray,
        ex_div_times: np.ndarray,
        q: float | None = None,  # Ignored, kept for interface compatibility
    ) -> float:
        if not call:
            raise NotImplementedError("Black's Approximation is implemented for calls only.")
        if discrete_dividends is None or ex_div_times is None or len(discrete_dividends) == 0:
            raise ValueError("BlacksApproxModel requires non-empty 'dividends' and 'div_times' arrays.")
        if len(discrete_dividends) != len(ex_div_times):
            raise ValueError("dividends and div_times must have the same length.")

        sigma = self.params["sigma"]

        # Method 1: Value of holding until maturity T
        pv_all_divs = sum(
            D * math.exp(-r * tD) for D, tD in zip(discrete_dividends, ex_div_times) if tD < t
        )
        S_adj_T = spot - pv_all_divs
        price_hold_to_maturity = self._bsm_price(S_adj_T, strike, r, t, sigma, call=True)

        # Method 2: Value of exercising just before an ex-dividend date
        prices_early_exercise = []
        for i, (D_i, t_i) in enumerate(zip(discrete_dividends, ex_div_times)):
            if t_i >= t:
                continue  # Skip dividends at or after maturity
            pv_divs_before_i = sum(
                discrete_dividends[j] * math.exp(-r * ex_div_times[j]) for j in range(i)
            )
            S_adj_i = spot - pv_divs_before_i
            price_at_t_i = self._bsm_price(S_adj_i, strike, r, t_i, sigma, call=True)
            prices_early_exercise.append(price_at_t_i)

        max_early_exercise_price = max(prices_early_exercise) if prices_early_exercise else 0.0
        return max(price_hold_to_maturity, max_early_exercise_price)

    def _bsm_price(
        self,
        spot: float,
        strike: float,
        r: float,
        t: float,
        sigma: float,
        call: bool = True,
    ) -> float:
        """Standard Black-Scholes formula (q=0 for simplicity)."""
        if spot <= 0 or t <= 0 or sigma <= 0:
            return max(0.0, spot - strike) if call else max(0.0, strike - spot)

        sqrt_t = np.sqrt(t)
        d1 = (np.log(spot / strike) + (r + 0.5 * sigma**2) * t) / (sigma * sqrt_t)
        d2 = d1 - sigma * sqrt_t
        df_rate = np.exp(-r * t)
        if call:
            return spot * norm.cdf(d1) - strike * df_rate * norm.cdf(d2)
        else:  # Included for completeness, though not used
            return strike * df_rate * norm.cdf(-d2) - spot * norm.cdf(-d1)

    def _cf_impl(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(f"{self.name} CF not implemented")

    def _sde_impl(self) -> Any:
        raise NotImplementedError(f"{self.name} SDE not implemented")

    def _pde_impl(self) -> Any:
        raise NotImplementedError(f"{self.name} PDE not implemented")