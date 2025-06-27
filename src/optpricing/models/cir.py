from __future__ import annotations

import math
from typing import Any

from optpricing.models.base import BaseModel, ParamValidator

__doc__ = """
Defines the Cox-Ingersoll-Ross (1985) mean-reverting interest rate model.
"""


class CIRModel(BaseModel):
    """
    Cox-Ingersoll-Ross (1985) mean-reverting short rate model.
    dr_t = kappa * (theta - r_t) * dt + sigma * sqrt(r_t) * dW_t
    """

    name: str = "Cox-Ingersoll-Ross"
    has_closed_form: bool = True
    default_params = {"kappa": 0.86, "theta": 0.09, "sigma": 0.02}

    def __init__(self, params: dict[str, float] | None = None):
        """
        Initializes the CIR model.

        Parameters
        ----------
        params : dict[str, float] | None, optional
            A dictionary of model parameters. If None, `default_params` are used.
        """
        super().__init__(params or self.default_params)

    def _validate_params(self) -> None:
        p = self.params
        req = ["kappa", "theta", "sigma"]
        ParamValidator.require(p, req, model=self.name)
        ParamValidator.positive(p, ["kappa", "theta", "sigma"], model=self.name)
        if 2 * p["kappa"] * p["theta"] < p["sigma"] ** 2:
            print("Warning: CIR parameters do not satisfy the Feller condition.")

    def _closed_form_impl(
        self,
        *,
        spot: float,
        t: float,
        **_: Any,
    ) -> float:
        """
        Calculates the price of a Zero-Coupon Bond.

        Note: Re-interprets 'spot' as the initial short rate r0 and 't' as
        the bond's maturity T. Ignores other option-specific parameters.

        Parameters
        ----------
        spot : float
            The initial short rate, r0.
        t : float
            The maturity of the bond, in years.

        Returns
        -------
        float
            The price of the zero-coupon bond.
        """
        r0, T = spot, t
        p = self.params
        kappa, theta, sigma = p["kappa"], p["theta"], p["sigma"]

        gamma = math.sqrt(kappa**2 + 2 * sigma**2)
        exp_gamma_T = math.exp(gamma * T)

        den = (gamma + kappa) * (exp_gamma_T - 1) + 2 * gamma
        B = 2 * (exp_gamma_T - 1) / den
        A_log_base = (2 * gamma * math.exp((kappa + gamma) * T / 2)) / den
        A_log_power = (2 * kappa * theta) / sigma**2

        price = (A_log_base**A_log_power) * math.exp(-B * r0)
        return price

    #  Abstract Method Implementations
    def _sde_impl(self, **kwargs: Any) -> Any:
        raise NotImplementedError

    def _cf_impl(self, **kwargs: Any) -> Any:
        raise NotImplementedError

    def _pde_impl(self, **kwargs: Any) -> Any:
        raise NotImplementedError
