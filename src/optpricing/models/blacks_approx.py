from __future__ import annotations

import math
from typing import Any

import numpy as np

from optpricing.models.base import BaseModel, ParamValidator
from optpricing.models.bsm import BSMModel

__doc__ = """
Defines Black's (1975) approximation for pricing an American call option
on a stock paying discrete dividends.
"""


class BlacksApproxModel(BaseModel):
    """
    Black's approximation for an American call on a stock with discrete dividends.
    """

    name: str = "Black's Approximation"
    has_closed_form: bool = True
    cf_kwargs = BaseModel.cf_kwargs + ("discrete_dividends", "ex_div_times")

    default_params = {"sigma": 0.30}

    def __init__(self, params: dict[str, float]):
        """
        Initializes the Black's Approximation model.

        Parameters
        ----------
        params : dict[str, float] | None, optional
            A dictionary of model parameters. If None, `default_params` are used.
            Must contain 'sigma'.
        """
        super().__init__(params)
        self.bsm_solver = BSMModel(params={"sigma": self.params["sigma"]})

    def _validate_params(self) -> None:
        """Validate the 'sigma' parameter."""
        ParamValidator.require(self.params, ["sigma"], model=self.name)
        ParamValidator.positive(self.params, ["sigma"], model=self.name)

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
        q: float | None = None,
    ) -> float:
        """
        Calculates the price by comparing holding vs. exercising before each dividend.

        Parameters
        ----------
        spot : float
            The current price of the underlying asset.
        strike : float
            The strike price of the option.
        r : float
            The continuously compounded risk-free rate.
        t : float
            The time to maturity of the option, in years.
        call : bool, optional
            Must be True, as the model is for calls only. Defaults to True.
        discrete_dividends : np.ndarray
            An array of discrete dividend amounts.
        ex_div_times : np.ndarray
            An array of ex-dividend dates, in years.

        Returns
        -------
        float
            The approximated price of the American call option.

        Raises
        ------
        NotImplementedError
            If the option is a put.
        ValueError
            If `discrete_dividends` is empty.
        """
        if not call:
            raise NotImplementedError(
                "Black's Approximation is for American calls only."
            )
        if not hasattr(discrete_dividends, "__len__") or len(discrete_dividends) == 0:
            raise ValueError(
                "BlacksApproxModel requires non-empty 'discrete_dividends'."
            )

        # Value of holding until maturity T
        pv_all_divs = sum(
            D * math.exp(-r * tD)
            for D, tD in zip(discrete_dividends, ex_div_times)
            if tD < t
        )
        S_adj_T = spot - pv_all_divs
        price_hold_to_maturity = self.bsm_solver.price_closed_form(
            spot=S_adj_T, strike=strike, r=r, q=0, t=t, call=True
        )

        # Value of exercising just before each ex-dividend date
        prices_early_exercise = []
        for i, t_i in enumerate(ex_div_times):
            if t_i >= t:
                continue
            pv_divs_before_i = sum(
                discrete_dividends[j] * math.exp(-r * ex_div_times[j]) for j in range(i)
            )
            S_adj_i = spot - pv_divs_before_i
            price_at_t_i = self.bsm_solver.price_closed_form(
                spot=S_adj_i, strike=strike, r=r, q=0, t=t_i, call=True
            )
            prices_early_exercise.append(price_at_t_i)

        max_early_price = max(prices_early_exercise) if prices_early_exercise else 0.0
        return max(price_hold_to_maturity, max_early_price)

    #  Abstract Method Implementations
    def _cf_impl(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def _sde_impl(self) -> Any:
        raise NotImplementedError

    def _pde_impl(self) -> Any:
        raise NotImplementedError
