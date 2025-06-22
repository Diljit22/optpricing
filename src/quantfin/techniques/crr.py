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


class CRRLatticeTechnique(BaseTechnique, GreekMixin, IVMixin, ABC):
    """
    Vectorized Cox-Ross-Rubinstein binomial (European/American).
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
        S0, K, T = stock.spot, option.strike, option.maturity
        r, q = rate.rate, stock.dividend
        sigma = model.params.get("sigma", stock.volatility)
        is_call = option.option_type is OptionType.CALL

        price = _crr_price(S0, K, T, r, q, sigma, self.steps, is_call, self.is_american)
        return PricingResult(price=price)

    def delta(
        self,
        option: Option,
        stock: Stock,
        model: Any,
        rate: Rate,
    ) -> float:
        S0, K, T = stock.spot, option.strike, option.maturity
        r, q = rate.rate, stock.dividend
        sigma = model.params.get("sigma", stock.volatility)
        is_call = option.option_type is OptionType.CALL

        dt = T / self.steps
        u = math.exp(sigma * math.sqrt(dt))
        d = 1.0 / u

        price_up = _crr_price(
            S0 * u, K, T, r, q, sigma, self.steps, is_call, self.is_american
        )
        price_dn = _crr_price(
            S0 * d, K, T, r, q, sigma, self.steps, is_call, self.is_american
        )

        return (price_up - price_dn) / (S0 * (u - d))

    def gamma(
        self,
        option: Option,
        stock: Stock,
        model: Any,
        rate: Rate,
    ) -> float:
        S0, K, T = stock.spot, option.strike, option.maturity
        r, q = rate.rate, stock.dividend
        sigma = model.params.get("sigma", stock.volatility)
        is_call = option.option_type is OptionType.CALL

        N = self.steps
        dt = T / N
        u = math.exp(sigma * math.sqrt(dt))
        d = 1.0 / u
        disc = math.exp(-r * dt)
        p = (math.exp((r - q) * dt) - d) / (u - d)
        p = max(0.0, min(1.0, p))  # clip to [0,1]

        j = np.arange(N + 1)
        ST = S0 * u**j * d ** (N - j)
        pay = np.where(is_call, np.maximum(ST - K, 0.0), np.maximum(K - ST, 0.0))

        for n in range(N - 1, 1, -1):  # stop when length = 3
            pay[:-1] = disc * (p * pay[1:] + (1 - p) * pay[:-1])

        Vuu, Vum, Vdd = pay[2], pay[1], pay[0]  # nodes at n = 2

        h_up = S0 * (u**2 - 1.0)
        h_dn = S0 * (1.0 - d**2)
        delta_up = (Vuu - Vum) / h_up
        delta_down = (Vum - Vdd) / h_dn
        avg_step = 0.5 * (S0 * u**2 - S0 * d**2)

        return (delta_up - delta_down) / avg_step


def _crr_price(
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
    dt = T / N
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    disc = math.exp(-r * dt)
    p = (math.exp((r - q) * dt) - d) / (u - d)
    p = min(max(p, 0.0), 1.0)

    j = np.arange(N + 1)
    ST = S0 * (u**j) * (d ** (N - j))
    pay = np.where(is_call, np.maximum(ST - K, 0.0), np.maximum(K - ST, 0.0))

    for n in range(N - 1, -1, -1):
        pay[:-1] = disc * (p * pay[1:] + (1 - p) * pay[:-1])
        if is_am:
            ST = ST[: n + 1] / u
            exc = np.where(is_call, np.maximum(ST - K, 0.0), np.maximum(K - ST, 0.0))
            pay[:-1] = np.maximum(pay[:-1], exc)
    return float(pay[0])
