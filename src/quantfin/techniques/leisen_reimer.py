from __future__ import annotations

import math
from abc import ABC
from typing import Any

import numpy as np

from quantfin.atoms.option import Option, OptionType
from quantfin.atoms.rate import Rate
from quantfin.atoms.stock import Stock
from quantfin.techniques.base.base_technique import BaseTechnique, PricingResult
from quantfin.techniques.base.greek_mixin import GreekMixin
from quantfin.techniques.base.iv_mixin import IVMixin


class LeisenReimerTechnique(BaseTechnique, GreekMixin, IVMixin, ABC):
    """
    Leisen-Reimer binomial via Peizer-Pratt (European/American).
    """

    def __init__(self, steps: int = 200, is_american: bool = False):
        super().__init__()
        self.steps = int(steps)
        self.is_american = bool(is_american)

    def price(
        self,
        option: Option,
        stock: Stock,
        model: Any,
        rate: Rate,
    ) -> PricingResult:
        S0 = stock.spot
        K = option.strike
        T = option.maturity
        r = rate.rate
        q = stock.dividend
        sigma = model.params.get("sigma", stock.volatility)
        call = option.option_type is OptionType.CALL

        price = _lr_price(S0, K, T, r, q, sigma, self.steps, call, self.is_american)
        return PricingResult(price=price)


def _peizer_pratt(z: float, N: int) -> float:
    if z == 0:  # Handle edge case to avoid division by zero
        return 0.5
    term = z / (N + 1 / 3 + 0.1 / (N + 1))
    return 0.5 + math.copysign(
        0.5 * math.sqrt(1 - math.exp(-(term**2) * (N + 1 / 6))), z
    )


def _lr_price(
    S0: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    N: int,
    is_call: bool,
    is_am: bool,
) -> float:
    # Ensure N is odd for Leisen-Reimer.
    if N % 2 == 0:
        N += 1

    dt = T / N
    disc = math.exp(-r * dt)
    cost_of_carry = r - q

    # Calculate d1 and d2 from Black-Scholes.
    sqrt_t = math.sqrt(T)
    if T <= 0 or sigma <= 0:
        # If time or vol is zero, return intrinsic value immediately.
        intrinsic = S0 * math.exp(-q * T) - K if is_call else K - S0 * math.exp(-q * T)
        return max(0.0, intrinsic * math.exp(-r * T))
    else:
        d1 = (math.log(S0 / K) + (cost_of_carry + 0.5 * sigma**2) * T) / (
            sigma * sqrt_t
        )
        d2 = d1 - sigma * sqrt_t

    p1 = _peizer_pratt(d1, N)
    p2 = _peizer_pratt(d2, N)

    # If p2 is ~0 or ~1, tree unstable; eps used to avoid floating point inaccuracies.
    epsilon = 1e-12
    if abs(p2) < epsilon or abs(1 - p2) < epsilon:
        # For European options, this is the exact BSM formula limit.
        # For American, it's a very strong approximation as early exercise is unlikely.
        if is_call:
            price = S0 * math.exp(-q * T) * p1 - K * math.exp(-r * T) * p2
        else:
            price = K * math.exp(-r * T) * (1 - p2) - S0 * math.exp(-q * T) * (1 - p1)
        return max(0.0, price)

    # Calculate the LR-specific up/down factors and the risk-neutral probability 'p'.
    p = p2
    u = math.exp(cost_of_carry * dt) * (p1 / p2)
    d = (math.exp(cost_of_carry * dt) - p * u) / (1 - p)

    # Valuation loop
    j = np.arange(N + 1)
    ST = S0 * (u**j) * (d ** (N - j))
    pay = np.where(is_call, np.maximum(ST - K, 0.0), np.maximum(K - ST, 0.0))

    for n in range(N - 1, -1, -1):
        pay[:-1] = disc * (p * pay[1:] + (1 - p) * pay[:-1])
        if is_am:
            j_inner = np.arange(n + 1)
            ST_inner = S0 * (u**j_inner) * (d ** (n - j_inner))
            exc = np.where(
                is_call, np.maximum(ST_inner - K, 0.0), np.maximum(K - ST_inner, 0.0)
            )
            pay[: n + 1] = np.maximum(pay[: n + 1], exc)

    return float(pay[0])
