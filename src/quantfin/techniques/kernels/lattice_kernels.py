from __future__ import annotations

import math
from abc import abstractmethod
from typing import Any

import numpy as np

from quantfin.atoms import Option, OptionType, Rate, Stock
from quantfin.models import BaseModel
from quantfin.techniques.base import BaseTechnique, GreekMixin, IVMixin, PricingResult


class LatticeTechnique(BaseTechnique, GreekMixin, IVMixin):
    """
    Abstract base class for lattice-based pricing techniques.

    This class provides a caching mechanism for Greek calculations. The main
    pricing method, `_price_and_get_nodes`, is designed to return both the
    option price and the values of the adjacent nodes at the first and second
    time steps, which are then used for instantaneous delta and gamma calculations
    without rebuilding the tree.
    """

    def __init__(self, steps: int = 200, is_american: bool = False):
        self.steps = int(steps)
        self.is_american = bool(is_american)
        self._cached_nodes: dict[str, Any] = {}

    def price(
        self, option: Option, stock: Stock, model: BaseModel, rate: Rate, **kwargs: Any
    ) -> PricingResult:
        """Prices the option and caches the necessary nodes for Greek calculations."""
        results = self._price_and_get_nodes(option, stock, model, rate)
        self._cached_nodes = results
        return PricingResult(price=results["price"])

    def delta(
        self, option: Option, stock: Stock, model: BaseModel, rate: Rate, **kwargs: Any
    ) -> float:
        """Calculates delta, using cached nodes if available."""
        if not self._cached_nodes:
            self.price(option, stock, model, rate)
        cache = self._cached_nodes
        return (cache["price_up"] - cache["price_down"]) / (
            cache["spot_up"] - cache["spot_down"]
        )

    def gamma(
        self, option: Option, stock: Stock, model: BaseModel, rate: Rate, **kwargs: Any
    ) -> float:
        """Calculates gamma, using cached nodes if available."""
        if not self._cached_nodes:
            self.price(option, stock, model, rate)
        cache = self._cached_nodes
        if "price_mid" in cache:  # Trinomial Case
            h1 = cache["spot_up"] - cache["spot_mid"]
            h2 = cache["spot_mid"] - cache["spot_down"]
            term1 = (cache["price_up"] - cache["price_mid"]) / h1
            term2 = (cache["price_mid"] - cache["price_down"]) / h2
            return 2 * (term1 - term2) / (h1 + h2)
        else:  # Binomial Case
            h_up = cache["spot_uu"] - cache["spot_ud"]
            h_down = cache["spot_ud"] - cache["spot_dd"]
            delta_up = (cache["price_uu"] - cache["price_ud"]) / h_up
            delta_down = (cache["price_ud"] - cache["price_dd"]) / h_down
            avg_spot_change = 0.5 * (cache["spot_uu"] - cache["spot_dd"])
            return (delta_up - delta_down) / avg_spot_change

    @abstractmethod
    def _price_and_get_nodes(
        self, option: Option, stock: Stock, model: BaseModel, rate: Rate
    ) -> dict[str, Any]:
        raise NotImplementedError


class CRRTechnique(LatticeTechnique):
    """Cox-Ross-Rubinstein binomial lattice technique."""

    def _price_and_get_nodes(
        self, option: Option, stock: Stock, model: BaseModel, rate: Rate
    ) -> dict[str, Any]:
        sigma = model.params.get("sigma", stock.volatility)
        return _crr_pricer(
            S0=stock.spot,
            K=option.strike,
            T=option.maturity,
            r=rate.get_rate(option.maturity),
            q=stock.dividend,
            sigma=sigma,
            N=self.steps,
            is_call=(option.option_type is OptionType.CALL),
            is_am=self.is_american,
        )


class LeisenReimerTechnique(LatticeTechnique):
    """Leisen-Reimer binomial lattice technique with Peizer-Pratt inversion."""

    def _price_and_get_nodes(
        self, option: Option, stock: Stock, model: BaseModel, rate: Rate
    ) -> dict[str, Any]:
        sigma = model.params.get("sigma", stock.volatility)
        return _lr_pricer(
            S0=stock.spot,
            K=option.strike,
            T=option.maturity,
            r=rate.get_rate(option.maturity),
            q=stock.dividend,
            sigma=sigma,
            N=self.steps,
            is_call=(option.option_type is OptionType.CALL),
            is_am=self.is_american,
        )


class TOPMTechnique(LatticeTechnique):
    """Kamrad-Ritchken trinomial lattice technique."""

    def _price_and_get_nodes(
        self, option: Option, stock: Stock, model: BaseModel, rate: Rate
    ) -> dict[str, Any]:
        sigma = model.params.get("sigma", stock.volatility)
        return _topm_pricer(
            S0=stock.spot,
            K=option.strike,
            T=option.maturity,
            r=rate.get_rate(option.maturity),
            q=stock.dividend,
            vol=sigma,
            N=self.steps,
            is_call=(option.option_type is OptionType.CALL),
            is_am=self.is_american,
        )


def _crr_pricer(
    S0: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    N: int,
    is_call: bool,
    is_am: bool,
) -> dict[str, Any]:
    if sigma < 1e-6:
        sigma = 1e-6
    dt = T / N
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    disc = math.exp(-r * dt)
    p = (math.exp((r - q) * dt) - d) / (u - d)
    j = np.arange(N + 1)
    payoff = np.where(
        is_call,
        np.maximum(S0 * u**j * d ** (N - j) - K, 0.0),
        np.maximum(K - S0 * u**j * d ** (N - j), 0.0),
    )
    for i in range(N - 1, -1, -1):
        if i == 1:
            price_uu, price_ud, price_dd = payoff[2], payoff[1], payoff[0]
        elif i == 0:
            price_up, price_down = payoff[1], payoff[0]
        payoff = disc * (p * payoff[1:] + (1 - p) * payoff[:-1])
        if is_am:
            stock_prices = S0 * u ** np.arange(i + 1) * d ** (i - np.arange(i + 1))
            early_exercise = np.where(
                is_call,
                np.maximum(stock_prices - K, 0.0),
                np.maximum(K - stock_prices, 0.0),
            )
            payoff = np.maximum(payoff, early_exercise)
    return {
        "price": payoff[0],
        "price_up": price_up,
        "price_down": price_down,
        "price_uu": price_uu,
        "price_ud": price_ud,
        "price_dd": price_dd,
        "spot_up": S0 * u,
        "spot_down": S0 * d,
        "spot_uu": S0 * u * u,
        "spot_ud": S0 * u * d,
        "spot_dd": S0 * d * d,
    }


def _lr_pricer(
    S0: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    N: int,
    is_call: bool,
    is_am: bool,
) -> dict[str, Any]:
    if sigma < 1e-6:
        price = max(0.0, S0 - K) if is_call else max(0.0, K - S0)
        return {
            "price": price,
            "price_up": price,
            "price_down": price,
            "price_uu": price,
            "price_ud": price,
            "price_dd": price,
            "spot_up": S0,
            "spot_down": S0,
            "spot_uu": S0,
            "spot_ud": S0,
            "spot_dd": S0,
        }
    if N % 2 == 0:
        N += 1
    dt = T / N
    disc = math.exp(-r * dt)
    d1 = (math.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    p_d1 = _peizer_pratt(d1, N)
    p_d2 = _peizer_pratt(d2, N)
    u = math.exp((r - q) * dt) * (p_d1 / p_d2)
    d = (math.exp((r - q) * dt) - p_d2 * u) / (1 - p_d2)
    p = p_d2
    j = np.arange(N + 1)
    payoff = np.where(
        is_call,
        np.maximum(S0 * u**j * d ** (N - j) - K, 0.0),
        np.maximum(K - S0 * u**j * d ** (N - j), 0.0),
    )
    for i in range(N - 1, -1, -1):
        if i == 1:
            price_uu, price_ud, price_dd = payoff[2], payoff[1], payoff[0]
        elif i == 0:
            price_up, price_down = payoff[1], payoff[0]
        payoff = disc * (p * payoff[1:] + (1 - p) * payoff[:-1])
        if is_am:
            stock_prices = S0 * u ** np.arange(i + 1) * d ** (i - np.arange(i + 1))
            early_exercise = np.where(
                is_call,
                np.maximum(stock_prices - K, 0.0),
                np.maximum(K - stock_prices, 0.0),
            )
            payoff = np.maximum(payoff, early_exercise)
    return {
        "price": payoff[0],
        "price_up": price_up,
        "price_down": price_down,
        "price_uu": price_uu,
        "price_ud": price_ud,
        "price_dd": price_dd,
        "spot_up": S0 * u,
        "spot_down": S0 * d,
        "spot_uu": S0 * u * u,
        "spot_ud": S0 * u * d,
        "spot_dd": S0 * d * d,
    }


def _peizer_pratt(z: float, N: int) -> float:
    if abs(z) > 50:
        return 1.0 if z > 0 else 0.0
    term = z / (N + 1 / 3 + 0.1 / (N + 1))
    return 0.5 + math.copysign(
        0.5 * math.sqrt(1 - math.exp(-(term**2) * (N + 1 / 6))), z
    )


def _topm_pricer(
    S0: float,
    K: float,
    T: float,
    r: float,
    q: float,
    vol: float,
    N: int,
    is_call: bool,
    is_am: bool,
) -> dict[str, Any]:
    if vol < 1e-6:
        vol = 1e-6
    dt = T / N
    disc = math.exp(-r * dt)
    dx = vol * math.sqrt(2 * dt)
    drift_term = (r - q - 0.5 * vol**2) * dt
    pu = 0.5 * ((vol**2 * dt + drift_term**2) / dx**2 + drift_term / dx)
    pd = 0.5 * ((vol**2 * dt + drift_term**2) / dx**2 - drift_term / dx)
    pm = 1.0 - pu - pd
    j = np.arange(-N, N + 1)
    payoff = np.where(
        is_call,
        np.maximum(S0 * np.exp(j * dx) - K, 0.0),
        np.maximum(K - S0 * np.exp(j * dx), 0.0),
    )
    for i in range(N - 1, -1, -1):
        if i == 0:
            price_up, price_mid, price_down = payoff[2], payoff[1], payoff[0]
        payoff = disc * (pu * payoff[2:] + pm * payoff[1:-1] + pd * payoff[:-2])
        if is_am:
            j_inner = np.arange(-i, i + 1)
            stock_prices = S0 * np.exp(j_inner * dx)
            early_exercise = np.where(
                is_call,
                np.maximum(stock_prices - K, 0.0),
                np.maximum(K - stock_prices, 0.0),
            )
            payoff = np.maximum(payoff, early_exercise)
    return {
        "price": payoff[0],
        "price_up": price_up,
        "price_mid": price_mid,
        "price_down": price_down,
        "spot_up": S0 * math.exp(dx),
        "spot_mid": S0,
        "spot_down": S0 * math.exp(-dx),
    }
