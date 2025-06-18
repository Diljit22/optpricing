from __future__ import annotations
from abc import ABC
import math
import numpy as np
from scipy.linalg import solve_banded

from quantfin.techniques.base.base_technique import BaseTechnique, PricingResult
from quantfin.techniques.base.greek_mixin    import GreekMixin
from quantfin.techniques.base.iv_mixin       import IVMixin
from quantfin.atoms.option                   import Option, OptionType
from quantfin.atoms.stock                    import Stock
from quantfin.atoms.rate                     import Rate
from quantfin.models.base                    import BaseModel
from quantfin.models.bsm                     import BSMModel


class PDESolverTechnique(BaseTechnique, GreekMixin, IVMixin, ABC):
    """
    Crank-Nicolson PDE solver for European options under BSM.
    Inherits GreekMixin & IVMixin, so all Greeks & IV work out of the box.
    """
    supports_pde = True

    def __init__(
        self,
        S_max: float = 500.0,
        M:     int   = 500,
        N:     int   = 500,
    ) -> None:
        super().__init__()
        self.S_max = float(S_max)
        self.M     = int(M)
        self.N     = int(N)

    def price(
        self,
        option: Option,
        stock:  Stock,
        model:  BaseModel,
        rate:   Rate,
    ) -> PricingResult:
        if not isinstance(model, BSMModel):
            raise TypeError(f"PDE technique only supported for BSMModel, got {model.__class__.__name__}")

        # unpack
        S0     = stock.spot
        K      = option.strike
        T      = option.maturity
        r      = rate.rate
        q      = stock.dividend
        sigma  = model.params["sigma"]

        M, N   = self.M, self.N
        dS     = self.S_max / M
        dt     = T / N

        # spatial grid and payoff
        S    = np.linspace(0.0, self.S_max, M+1)
        call = (option.option_type is OptionType.CALL)
        V    = np.maximum(S-K, 0.0) if call else np.maximum(K-S, 0.0)

        # prepare coefficients
        S_int     = S[1:-1]
        a         = 0.5*sigma**2 * S_int**2 / dS**2
        b         = (r-q)*S_int / (2.0*dS)
        c         = r * np.ones_like(S_int)
        theta     = 0.5  # Crank-Nicolson

        alpha     = -theta*dt*(a - b)
        beta      =  1.0 + theta*dt*(2.0*a + c)
        gamma     = -theta*dt*(a + b)

        ab        = np.zeros((3, M-1))
        ab[0,1:]  = gamma[:-1]
        ab[1,:]   = beta
        ab[2,:-1] = alpha[1:]

        alpha_e   = (1-theta)*dt*(a - b)
        beta_e    = 1.0 - (1-theta)*dt*(2.0*a + c)
        gamma_e   = (1-theta)*dt*(a + b)

        # time‐step backwards
        for n in range(N):
            t = T - n*dt
            # boundaries
            if call:
                V[0]  = 0.0
                V[-1] = max(0.0, S[-1]*math.exp(-q*t) - K*math.exp(-r*t))
            else:
                V[0]  = max(0.0, K*math.exp(-r*t))
                V[-1] = 0.0

            V_int = V[1:-1]
            rhs   = np.empty(M-1)
            # explicit RHS
            rhs[0]    = alpha_e[0]*V[0] + beta_e[0]*V_int[0] + gamma_e[0]*V_int[1]
            for i in range(1, M-2):
                rhs[i] = (
                    alpha_e[i]*V_int[i-1]
                    + beta_e[i]*V_int[i]
                    + gamma_e[i]*V_int[i+1]
                )
            rhs[-1]   = alpha_e[-1]*V_int[-2] + beta_e[-1]*V_int[-1] + gamma_e[-1]*V[-1]

            V[1:-1] = solve_banded((1,1), ab, rhs)

        # interpolate to S0
        if S0 <= 0:
            price = V[0]
        elif S0 >= self.S_max:
            price = V[-1]
        else:
            j      = min(max(int(S0/dS), 0), M-1)
            weight = (S0 - S[j]) / dS
            price  = V[j] + weight*(V[j+1] - V[j])

        return PricingResult(price=price)

    def gamma(self, option, stock, model, rate, **kwargs):
        if hasattr(model, "gamma_analytic"):
            return model.gamma_analytic(
                spot=   stock.spot,
                strike= option.strike,
                r=      rate.rate,
                q=      stock.dividend,
                t=      option.maturity,
            )
        return super().gamma(option, stock, model, rate, **kwargs)