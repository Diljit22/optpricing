from __future__ import annotations

from typing import Any

import numba
import numpy as np

from optpricing.atoms import Option, OptionType, Rate, Stock
from optpricing.models import BaseModel, BSMModel
from optpricing.techniques.base import BaseTechnique, GreekMixin, IVMixin, PricingResult

__doc__ = """
Defines a pricing technique based on solving the Black-Scholes Partial
Differential Equation (PDE) using a finite difference method.
"""


@numba.njit(cache=True)
def _thomas_factors(lower, diag, upper):
    """Forward-elimination factors of a constant tridiagonal matrix
    (sub-diagonal `lower[1:]`, diagonal `diag`, super-diagonal `upper[:-1]`)."""
    n = diag.shape[0]
    c_prime = np.empty(n)
    denom = np.empty(n)
    denom[0] = diag[0]
    c_prime[0] = upper[0] / denom[0]
    for i in range(1, n):
        denom[i] = diag[i] - lower[i] * c_prime[i - 1]
        c_prime[i] = upper[i] / denom[i] if i < n - 1 else 0.0
    return c_prime, denom


@numba.njit(cache=True)
def _thomas_solve(lower, c_prime, denom, rhs, out):
    n = rhs.shape[0]
    out[0] = rhs[0] / denom[0]
    for i in range(1, n):
        out[i] = (rhs[i] - lower[i] * out[i - 1]) / denom[i]
    for i in range(n - 2, -1, -1):
        out[i] -= c_prime[i] * out[i + 1]


@numba.njit(cache=True)
def _march(V, a, b, c, thetas, hs, v_low, v_high):
    """
    Time-march the theta scheme (I - theta h A) V_new = (I + (1-theta) h A) V_old
    with Dirichlet boundaries `v_low[s]`, `v_high[s]` at sub-step s. Each distinct
    (theta, h) pair is factorised once.
    """
    n = a.shape[0]
    n_sub = thetas.shape[0]
    rhs = np.empty(n)
    inner = np.empty(n)
    # at most two distinct (theta, h) pairs: implicit half-steps and CN steps
    th_a, h_a = thetas[0], hs[0]
    lower_a = -th_a * h_a * a
    diag_a = 1.0 - th_a * h_a * b
    upper_a = -th_a * h_a * c
    cp_a, den_a = _thomas_factors(lower_a, diag_a, upper_a)
    th_b, h_b = thetas[n_sub - 1], hs[n_sub - 1]
    lower_b = -th_b * h_b * a
    diag_b = 1.0 - th_b * h_b * b
    upper_b = -th_b * h_b * c
    cp_b, den_b = _thomas_factors(lower_b, diag_b, upper_b)
    for s in range(n_sub):
        theta, h = thetas[s], hs[s]
        explicit = (1.0 - theta) * h
        for j in range(n):
            rhs[j] = V[j + 1] + explicit * (
                a[j] * V[j] + b[j] * V[j + 1] + c[j] * V[j + 2]
            )
        rhs[0] += theta * h * a[0] * v_low[s]
        rhs[n - 1] += theta * h * c[n - 1] * v_high[s]
        if theta == th_a and h == h_a:
            _thomas_solve(lower_a, cp_a, den_a, rhs, inner)
        else:
            _thomas_solve(lower_b, cp_b, den_b, rhs, inner)
        for j in range(n):
            V[j + 1] = inner[j]
        V[0] = v_low[s]
        V[n + 1] = v_high[s]
    return V


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
        self._cache_key: tuple | None = None

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

        def boundary(tau: float) -> tuple[float, float]:
            """Dirichlet values at S=0 and S=S_max for time-to-expiry tau."""
            if is_call:
                return 0.0, S_max * np.exp(-q * tau) - K * np.exp(-r * tau)
            return K * np.exp(-r * tau), 0.0

        V = np.maximum(S_vec - K, 0.0) if is_call else np.maximum(K - S_vec, 0.0)

        # Rannacher start-up: the first `rannacher_steps` CN steps are each
        # replaced by two fully implicit half-steps.
        n_r = min(self.rannacher_steps, N)
        thetas = np.concatenate([np.ones(2 * n_r), np.full(N - n_r, 0.5)])
        hs = np.concatenate([np.full(2 * n_r, 0.5 * dt), np.full(N - n_r, dt)])
        taus = np.cumsum(hs)
        bounds = np.array([boundary(t) for t in taus])
        V = _march(
            V,
            a,
            b,
            c,
            thetas,
            hs,
            np.ascontiguousarray(bounds[:, 0]),
            np.ascontiguousarray(bounds[:, 1]),
        )

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
        self._cache_key = (option, stock, model, rate)
        return PricingResult(price=self._cached_results["price"])

    def _ensure_cached(
        self, option: Option, stock: Stock, model: BaseModel, rate: Rate
    ) -> None:
        if self._cache_key != (option, stock, model, rate):
            self.price(option, stock, model, rate)

    def delta(
        self,
        option: Option,
        stock: Stock,
        model: BaseModel,
        rate: Rate,
        **kwargs,
    ) -> float:
        """
        Returns the delta from the PDE grid.

        The grid of the last `price` call is reused if it was built for the
        same inputs; otherwise the option is priced first.
        """
        self._ensure_cached(option, stock, model, rate)
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
        Returns the gamma from the PDE grid.

        The grid of the last `price` call is reused if it was built for the
        same inputs; otherwise the option is priced first.
        """
        self._ensure_cached(option, stock, model, rate)
        return self._cached_results["gamma"]
