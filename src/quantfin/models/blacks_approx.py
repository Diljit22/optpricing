from __future__ import annotations
import math
import numpy as np
from typing import Optional, Any

from quantfin.models.base import BaseModel, ParamValidator
from quantfin.models.bsm import BSMModel

class BlacksApproxModel(BaseModel):
    """
    Black's (1975) approximation for an American call on a stock with discrete dividends.
    """
    name: str = "Black's Approximation"
    has_closed_form: bool = True

    def __init__(self, params: dict[str, float]):
        super().__init__(params)
        # Composition: Create a BSMModel instance to handle the core pricing.
        self.bsm_solver = BSMModel(params={"sigma": self.params["sigma"]})

    def _validate_params(self) -> None:
        ParamValidator.require(self.params, ["sigma"], model=self.name)
        ParamValidator.positive(self.params, ["sigma"], model=self.name)

    def _closed_form_impl(self, *, spot: float, strike: float, r: float, t: float, call: bool = True, discrete_dividends: np.ndarray, ex_div_times: np.ndarray, q: float | None = None) -> float:
        if not call:
            raise NotImplementedError("Black's Approximation is for American calls only.")
        if not hasattr(discrete_dividends, '__len__') or len(discrete_dividends) == 0:
            raise ValueError("BlacksApproxModel requires non-empty 'discrete_dividends'.")

        # Method 1: Value of holding until maturity T
        pv_all_divs = sum(D * math.exp(-r * tD) for D, tD in zip(discrete_dividends, ex_div_times) if tD < t)
        S_adj_T = spot - pv_all_divs
        price_hold_to_maturity = self.bsm_solver.price_closed_form(spot=S_adj_T, strike=strike, r=r, q=0, t=t, call=True)

        # Method 2: Value of exercising just before each ex-dividend date
        prices_early_exercise = []
        for i, t_i in enumerate(ex_div_times):
            if t_i >= t:
                continue
            pv_divs_before_i = sum(discrete_dividends[j] * math.exp(-r * ex_div_times[j]) for j in range(i))
            S_adj_i = spot - pv_divs_before_i
            price_at_t_i = self.bsm_solver.price_closed_form(spot=S_adj_i, strike=strike, r=r, q=0, t=t_i, call=True)
            prices_early_exercise.append(price_at_t_i)

        max_early_price = max(prices_early_exercise) if prices_early_exercise else 0.0
        return max(price_hold_to_maturity, max_early_price)

    def _cf_impl(self, *args: Any, **kwargs: Any) -> Any: raise NotImplementedError
    def _sde_impl(self) -> Any: raise NotImplementedError
    def _pde_impl(self) -> Any: raise NotImplementedError