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


class TOPMLatticeTechnique(BaseTechnique, GreekMixin, IVMixin, ABC):
    """
    Kamrad-Ritchken recombining trinomial (European/American), log-space grid.
    """
    def __init__(self, steps: int = 101, is_american: bool = False):
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
        vol = model.params.get("sigma", stock.volatility)
        call = (option.option_type is OptionType.CALL)

        price = _topm_price(S0, K, T, r, q, vol, self.steps, call, self.is_american)
        return PricingResult(price=price)


    def delta(
        self,
        option: Option,
        stock:  Stock,
        model:  Any,
        rate:   Rate,
    ) -> float:
        S0, K, T = stock.spot, option.strike, option.maturity
        r, q     = rate.rate, stock.dividend
        vol      = model.params.get("sigma", stock.volatility)
        is_call  = option.option_type is OptionType.CALL

        dT  = T / self.steps
        noise = vol * math.sqrt(dT / 2.0)
        B   = math.exp(2.0 * noise)

        V_up = _topm_price(S0 * B,   K, T, r, q, vol,
                           self.steps, is_call, self.is_american)
        V_dn = _topm_price(S0 / B,   K, T, r, q, vol,
                           self.steps, is_call, self.is_american)

        return (V_up - V_dn) / (S0 * (B - 1.0/B))


    def gamma(
        self,
        option: Option,
        stock:  Stock,
        model:  Any,
        rate:   Rate,
    ) -> float:
        S0, K, T = stock.spot, option.strike, option.maturity
        r, q     = rate.rate, stock.dividend
        vol      = model.params.get("sigma", stock.volatility)
        is_call  = option.option_type is OptionType.CALL

        dT  = T / self.steps
        noise = vol * math.sqrt(dT / 2.0)
        B   = math.exp(2.0 * noise)

        V_up  = _topm_price(S0 * B,   K, T, r, q, vol,
                            self.steps, is_call, self.is_american)
        V_mid = _topm_price(S0,       K, T, r, q, vol,
                            self.steps, is_call, self.is_american)
        V_dn  = _topm_price(S0 / B,   K, T, r, q, vol,
                            self.steps, is_call, self.is_american)

        h1 = S0 * (B - 1.0)
        h2 = S0 * (1.0 - 1.0/B)

        return (
            2.0 * (h2 * V_up - (h1 + h2) * V_mid + h1 * V_dn)
            / (h1 * h2 * (h1 + h2))
        )


def _topm_price(
    S0: float, K: float, T: float,
    r: float, q: float, vol: float,
    N: int, is_call: bool, is_am: bool
) -> float:
    dT = T / N
    disc = math.exp(-r * dT)
    dT2 = dT / 2
    drift = (r - q) * dT2
    noise = vol * math.sqrt(dT2)

    B = math.exp(2 * noise)
    A = math.exp(drift + noise)
    norm = (B - 1)**2
    pU = (A - 1)**2 / norm
    pD = (B - A)**2 / norm
    pS = 1.0 - pU - pD
    pU = min(max(pU, 0.0), 1.0)
    pD = min(max(pD, 0.0), 1.0)
    pS = min(max(pS, 0.0), 1.0)

    j = np.arange(-N, N + 1)
    S_vec = S0 * (B**j)
    pay = np.where(is_call, np.maximum(S_vec - K, 0.0), np.maximum(K - S_vec, 0.0))

    for i in range(N - 1, -1, -1):
        M = 2 * i + 1
        pay[:M] = disc * (
            pU * pay[2 : M + 2] + pS * pay[1 : M + 1] + pD * pay[:M]
        )
        if is_am:
            idx = N - i
            cont = np.where(
                is_call,
                np.maximum(S_vec[idx : idx + M] - K, 0.0),
                np.maximum(K - S_vec[idx : idx + M], 0.0),
            )
            pay[:M] = np.maximum(pay[:M], cont)
        else:
            pay[:M] = np.maximum(pay[:M], 0.0)

    return float(pay[0])
