from __future__ import annotations

from typing import Any

import numpy as np
from scipy.linalg import solve_banded

from optpricing.atoms import Option, OptionType, Rate, Stock
from optpricing.models import BaseModel, BSMModel
from optpricing.techniques.base import BaseTechnique, GreekMixin, IVMixin, PricingResult

__doc__ = """
Defines a pricing technique based on solving the Black-Scholes Partial
Differential Equation (PDE) using a finite difference method.
"""


class PDETechnique(BaseTechnique, GreekMixin, IVMixin):
    """
    Prices options by solving the Black-Scholes PDE with a Crank-Nicolson scheme.

    This technique is optimized for the BSM model and calculates the price, delta,
    and gamma in a single pass by building a grid of asset prices and time steps.
    """

    def __init__(
        self,
        S_max_mult: float = 3.0,
        M: int = 200,
        N: int = 200,
        rannacher_steps: int = 2,
    ):
        """
        Initializes the PDE solver.

        Parameters
        ----------
        S_max_mult : float, optional
            Multiplier for the initial spot price to set the maximum grid boundary,
            by default 3.0. The grid spacing is chosen so that the spot lies
            exactly on a node.
        M : int, optional
            Number of asset price steps (grid columns), by default 200.
        N : int, optional
            Number of time steps (grid rows), by default 200.
        rannacher_steps : int, optional
            Number of initial Crank-Nicolson steps replaced by pairs of fully
            implicit half-steps (Rannacher 1984 start-up). This damps the
            oscillations the non-smooth payoff otherwise induces in gamma.
            Default 2; set 0 for plain Crank-Nicolson.
        """
        self.S_max_mult = float(S_max_mult)
        self.M = int(M)
        self.N = int(N)
        self.rannacher_steps = int(rannacher_steps)
        self._cached_results: dict[str, Any] = {}

    def _price_and_greeks(
        self,
        option: Option,
        stock: Stock,
        model: BaseModel,
        rate: Rate,
    ) -> dict:
        """
        Internal method to run the Crank-Nicolson solver and extract results.
        """
        if not isinstance(model, BSMModel):
            raise TypeError(
                f"PDETechnique is optimized for BSMModel only, but got {model.name}."
            )
        S0, K, T = stock.spot, option.strike, option.maturity
        r, q, sigma = rate.get_rate(T), stock.dividend, model.params["sigma"]
        M, N = self.M, self.N
        is_call = option.option_type is OptionType.CALL

        # Place the spot exactly on node j0 so that the grid Greeks are taken
        # at S0 rather than at the nearest node below it.
        j0 = max(2, min(M - 2, round(M / self.S_max_mult)))
        dS = S0 / j0
        S_max = M * dS
        S_vec = dS * np.arange(M + 1)
        dt = T / N

        # Spatial operator A (backward time): dV/dtau = a V_{j-1} + b V_j + c V_{j+1}
        j = np.arange(1, M)
        a = 0.5 * (sigma**2 * j**2 - (r - q) * j)
        b = -(sigma**2 * j**2 + r)
        c = 0.5 * (sigma**2 * j**2 + (r - q) * j)

        def banded(theta: float, h: float) -> np.ndarray:
            """Banded matrix of (I - theta*h*A) for scipy.linalg.solve_banded."""
            lhs = np.zeros((3, M - 1))
            lhs[0, 1:] = -theta * h * c[:-1]
            lhs[1, :] = 1 - theta * h * b
            lhs[2, :-1] = -theta * h * a[1:]
            return lhs

        def boundary(tau: float) -> tuple[float, float]:
            """Dirichlet values at S=0 and S=S_max for time-to-expiry tau."""
            if is_call:
                return 0.0, S_max * np.exp(-q * tau) - K * np.exp(-r * tau)
            return K * np.exp(-r * tau), 0.0

        V = np.maximum(S_vec - K, 0.0) if is_call else np.maximum(K - S_vec, 0.0)

        # Rannacher start-up: the first `rannacher_steps` CN steps are each
        # replaced by two fully implicit half-steps.
        schedule: list[tuple[float, float]] = []  # (theta, h) per sub-step
        n_r = min(self.rannacher_steps, N)
        schedule += [(1.0, 0.5 * dt)] * (2 * n_r)
        schedule += [(0.5, dt)] * (N - n_r)

        lhs_cache: dict[tuple[float, float], np.ndarray] = {}
        tau = 0.0
        for theta, h in schedule:
            lhs = lhs_cache.get((theta, h))
            if lhs is None:
                lhs = lhs_cache[(theta, h)] = banded(theta, h)
            explicit = 1.0 - theta
            rhs = V[1:-1] + explicit * h * (a * V[:-2] + b * V[1:-1] + c * V[2:])
            tau += h
            v0_new, vM_new = boundary(tau)
            rhs[0] += theta * h * a[0] * v0_new
            rhs[-1] += theta * h * c[-1] * vM_new
            V[1:-1] = solve_banded((1, 1), lhs, rhs, overwrite_b=True)
            V[0], V[-1] = v0_new, vM_new

        delta = (V[j0 + 1] - V[j0 - 1]) / (2 * dS)
        gamma_val = (V[j0 + 1] - 2 * V[j0] + V[j0 - 1]) / (dS**2)

        return {
            "price": float(V[j0]),
            "delta": float(delta),
            "gamma": float(gamma_val),
        }

    def price(
        self,
        option: Option,
        stock: Stock,
        model: BaseModel,
        rate: Rate,
        **kwargs,
    ) -> PricingResult:
        """
        Calculates the option price and caches grid-based Greeks.

        Parameters
        ----------
        option : Option
            The option contract to be priced.
        stock : Stock
            The underlying asset's properties.
        model : BaseModel
            The financial model to use. Must be a BSMModel.
        rate : Rate
            The risk-free rate structure.

        Returns
        -------
        PricingResult
            An object containing the calculated price.
        """
        self._cached_results = self._price_and_greeks(option, stock, model, rate)
        return PricingResult(price=self._cached_results["price"])

    def delta(
        self,
        option: Option,
        stock: Stock,
        model: BaseModel,
        rate: Rate,
        **kwargs,
    ) -> float:
        """
        Returns the cached delta from the PDE grid.

        If the cache is empty, it first runs the pricing calculation.
        """
        if not self._cached_results:
            self.price(option, stock, model, rate)
        return self._cached_results["delta"]

    def gamma(
        self,
        option: Option,
        stock: Stock,
        model: BaseModel,
        rate: Rate,
        **kwargs,
    ) -> float:
        """
        Returns the cached gamma from the PDE grid.

        If the cache is empty, it first runs the pricing calculation.
        """
        if not self._cached_results:
            self.price(option, stock, model, rate)
        return self._cached_results["gamma"]
