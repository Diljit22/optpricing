from __future__ import annotations

from typing import Any

import numpy as np
from scipy.linalg import solve_banded

from quantfin.atoms import Option, OptionType, Rate, Stock
from quantfin.models import BaseModel, BSMModel
from quantfin.techniques.base import BaseTechnique, GreekMixin, IVMixin, PricingResult


class PDETechnique(BaseTechnique, GreekMixin, IVMixin):
    def __init__(self, S_max_mult: float = 3.0, M: int = 200, N: int = 200):
        self.S_max_mult = float(S_max_mult)
        self.M = int(M)
        self.N = int(N)
        self._cached_results: dict[str, Any] = {}

    def _price_and_greeks(self, option: Option, stock: Stock, model: BaseModel, rate: Rate) -> dict:
        if not isinstance(model, BSMModel):
            raise TypeError(f"This PDETechnique is optimized for BSMModel only, but got {model.name}.")
        S0, K, T = stock.spot, option.strike, option.maturity
        r, q, sigma = rate.get_rate(T), stock.dividend, model.params["sigma"]
        S_max = S0 * self.S_max_mult
        M, N = self.M, self.N
        dS, dt = S_max / M, T / N
        S_vec = np.linspace(0, S_max, M + 1)
        is_call = (option.option_type is OptionType.CALL)
        j = np.arange(1, M)
        alpha = 0.25 * dt * (sigma**2 * j**2 - (r - q) * j)
        beta = -0.5 * dt * (sigma**2 * j**2 + r)
        gamma = 0.25 * dt * (sigma**2 * j**2 + (r - q) * j)
        LHS = np.zeros((3, M - 1))
        LHS[0, 1:], LHS[1, :], LHS[2, :-1] = -gamma[:-1], 1 - beta, -alpha[1:]
        V = np.maximum(S_vec - K, 0) if is_call else np.maximum(K - S_vec, 0)
        for i in range(1, N + 1):
            rhs = alpha * V[:-2] + (1 + beta) * V[1:-1] + gamma * V[2:]
            time_to_expiry = T - i * dt
            if is_call:
                rhs[-1] += gamma[-1] * (S_max - K * np.exp(-r * time_to_expiry))
            else:
                rhs[0] += alpha[0] * (K * np.exp(-r * time_to_expiry))
            V[1:-1] = solve_banded((1, 1), LHS, rhs, overwrite_b=True)

        # Grid-based Greeks
        j0 = int(S0 / dS)
        delta = (V[j0 + 1] - V[j0 - 1]) / (2 * dS)
        gamma_val = (V[j0 + 1] - 2 * V[j0] + V[j0 - 1]) / (dS**2)

        return {
            "price": np.interp(S0, S_vec, V),
            "delta": delta,
            "gamma": gamma_val,
        }

    def price(self, option: Option, stock: Stock, model: BaseModel, rate: Rate, **kwargs) -> PricingResult:
        self._cached_results = self._price_and_greeks(option, stock, model, rate)
        return PricingResult(price=self._cached_results['price'])

    def delta(self, option: Option, stock: Stock, model: BaseModel, rate: Rate, **kwargs) -> float:
        if not self._cached_results: self.price(option, stock, model, rate)
        return self._cached_results['delta']

    def gamma(self, option: Option, stock: Stock, model: BaseModel, rate: Rate, **kwargs) -> float:
        if not self._cached_results: self.price(option, stock, model, rate)
        return self._cached_results['gamma']
