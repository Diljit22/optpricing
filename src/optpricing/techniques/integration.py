from __future__ import annotations

from typing import Any

import numpy as np
from scipy import integrate

from optpricing.atoms import Option, OptionType, Rate, Stock
from optpricing.models import BaseModel
from optpricing.techniques.base import BaseTechnique, GreekMixin, IVMixin, PricingResult

__doc__ = """
Defines a pricing technique based on the Gil-Pelaez inversion formula,
solved using numerical quadrature.
"""


class IntegrationTechnique(BaseTechnique, GreekMixin, IVMixin):
    """
    Prices options using the Gil-Pelaez inversion formula via numerical quadrature.

    This technique leverages the model's characteristic function (CF) to price
    options. It is particularly useful for models where a closed-form solution
    is unavailable but the CF is known (e.g., Heston, Bates, VG, NIG).

    It provides an analytic delta as a "free" byproduct of the pricing calculation.
    """

    _node_cache: dict[tuple[float, int], tuple[np.ndarray, np.ndarray]] = {}

    def __init__(
        self,
        *,
        upper_bound: float = 200.0,
        limit: int = 200,
        epsabs: float = 1e-9,
        epsrel: float = 1e-9,
        method: str = "gauss",
        n_nodes: int = 24,
    ):
        """
        Initializes the numerical integration solver.

        Parameters
        ----------
        upper_bound : float, optional
            The upper limit of the integration, by default 200.0.
        limit : int, optional
            Maximum number of sub-intervals for the adaptive method, by default 200.
        epsabs : float, optional
            Absolute error tolerance of the adaptive method, by default 1e-9.
        epsrel : float, optional
            Relative error tolerance of the adaptive method, by default 1e-9.
        method : {"gauss", "quad"}, optional
            ``"gauss"`` (default) evaluates the characteristic function once on a
            fixed composite Gauss-Legendre rule (geometric panels
            ``[0,1],[1,2],[2,4],...`` up to ``upper_bound``, ``n_nodes`` per
            panel), which agrees with the adaptive rule to ~1e-13 for the
            library's models at a fraction of the cost. ``"quad"`` uses
            ``scipy.integrate.quad`` with a Python callback per node.
        n_nodes : int, optional
            Gauss-Legendre nodes per panel, by default 24.
        """
        if method not in ("gauss", "quad"):
            raise ValueError("method must be 'gauss' or 'quad'.")
        self.upper_bound = upper_bound
        self.limit = limit
        self.epsabs = epsabs
        self.epsrel = epsrel
        self.method = method
        self.n_nodes = int(n_nodes)
        self._cached_results: dict[str, Any] = {}

    @classmethod
    def gauss_nodes(
        cls, upper_bound: float, n_nodes: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Composite Gauss-Legendre nodes and weights on geometric panels of
        ``[0, upper_bound]`` (cached)."""
        key = (float(upper_bound), int(n_nodes))
        cached = cls._node_cache.get(key)
        if cached is None:
            edges = [0.0, 1.0]
            while edges[-1] * 2.0 < upper_bound:
                edges.append(edges[-1] * 2.0)
            edges.append(float(upper_bound))
            x, w = np.polynomial.legendre.leggauss(n_nodes)
            nodes, weights = [], []
            for a, b in zip(edges[:-1], edges[1:]):
                nodes.append(0.5 * (b - a) * x + 0.5 * (b + a))
                weights.append(0.5 * (b - a) * w)
            cached = cls._node_cache[key] = (
                np.concatenate(nodes),
                np.concatenate(weights),
            )
        return cached

    def _price_and_delta(
        self,
        option: Option,
        stock: Stock,
        model: BaseModel,
        rate: Rate,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Internal method to perform the core calculation once.

        Parameters
        ----------
        option : Option
            The option contract to be priced.
        stock : Stock
            The underlying asset's properties.
        model : BaseModel
            The financial model to use. Must support a characteristic function.
        rate : Rate
            The risk-free rate structure.
        """
        if not model.supports_cf:
            raise TypeError(
                f"Model '{model.name}' does not support a characteristic function."
            )

        S, K, T = stock.spot, option.strike, option.maturity
        r, q = rate.get_rate(T), stock.dividend
        phi = model.cf(t=T, spot=S, r=r, q=q, **kwargs)
        k_log = np.log(K)

        phi_minus_i = phi(-1j)
        if self.method == "gauss":
            u, w = self.gauss_nodes(self.upper_bound, self.n_nodes)
            twist = np.exp(-1j * u * k_log) / u
            integral_p2 = float(np.dot(w, (twist * phi(u)).imag))
            if np.abs(phi_minus_i) < 1e-12:
                P1 = np.nan
            else:
                integral_p1 = float(np.dot(w, (twist * phi(u - 1j)).imag))
                P1 = 0.5 + integral_p1 / (np.pi * np.real(phi_minus_i))
        else:

            def integrand_p2(u):
                """Calculates the integrand for P2."""
                return (np.exp(-1j * u * k_log) * phi(u)).imag / u

            def integrand_p1(u):
                """Calculates the integrand for P1."""
                return (np.exp(-1j * u * k_log) * phi(u - 1j)).imag / u

            integral_p2, _ = integrate.quad(
                integrand_p2,
                1e-15,
                self.upper_bound,
                limit=self.limit,
                epsabs=self.epsabs,
                epsrel=self.epsrel,
            )
            if np.abs(phi_minus_i) < 1e-12:
                P1 = np.nan
            else:
                integral_p1, _ = integrate.quad(
                    integrand_p1,
                    1e-15,
                    self.upper_bound,
                    limit=self.limit,
                    epsabs=self.epsabs,
                    epsrel=self.epsrel,
                )
                P1 = 0.5 + integral_p1 / (np.pi * np.real(phi_minus_i))

        P2 = 0.5 + integral_p2 / np.pi

        if np.isnan(P1):
            price, delta = np.nan, np.nan
        else:
            df_S = S * np.exp(-q * T)
            df_K = K * np.exp(-r * T)
            if option.option_type is OptionType.CALL:
                price = df_S * P1 - df_K * P2
                delta = np.exp(-q * T) * P1
            else:
                price = df_K * (1 - P2) - df_S * (1 - P1)
                delta = np.exp(-q * T) * (P1 - 1.0)

        return {"price": price, "delta": delta}

    def price(
        self,
        option: Option,
        stock: Stock,
        model: BaseModel,
        rate: Rate,
        **kwargs: Any,
    ) -> PricingResult:
        """
        Calculates the option price and caches the 'free' analytic delta.

        Parameters
        ----------
        option : Option
            The option contract to be priced.
        stock : Stock
            The underlying asset's properties.
        model : BaseModel
            The financial model to use. Must support a characteristic function.
        rate : Rate
            The risk-free rate structure.

        Returns
        -------
        PricingResult
            An object containing the calculated price.
        """
        self._cached_results = self._price_and_delta(
            option, stock, model, rate, **kwargs
        )
        return PricingResult(price=self._cached_results["price"])

    def delta(
        self,
        option: Option,
        stock: Stock,
        model: BaseModel,
        rate: Rate,
        **kwargs: Any,
    ) -> float:
        """
        Returns the 'free' delta calculated during the pricing call.

        If the cache is empty or the analytic delta calculation failed, it
        falls back to the numerical finite difference method from `GreekMixin`.

        Parameters
        ----------
        option : Option
            The option contract to be priced.
        stock : Stock
            The underlying asset's properties.
        model : BaseModel
            The financial model to use. Must support a characteristic function.
        rate : Rate
            The risk-free rate structure.

        Returns
        -------
        float
            Delta of the option.
        """
        if not self._cached_results:
            self.price(option, stock, model, rate, **kwargs)

        delta_val = self._cached_results.get("delta")
        if delta_val is not None and not np.isnan(delta_val):
            return delta_val
        else:
            # Fallback to finite difference if analytic delta failed
            return super().delta(option, stock, model, rate, **kwargs)
