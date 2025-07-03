from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np

from optpricing.atoms import Option, OptionType, Rate, Stock
from optpricing.models import BaseModel, SABRJumpModel, SABRModel
from optpricing.techniques.base import BaseTechnique, GreekMixin, IVMixin, PricingResult

from .kernels.american_mc_kernels import longstaff_schwartz_pricer
from .kernels.path_kernels import (
    bates_path_kernel,
    bsm_path_kernel,
    heston_path_kernel,
    kou_path_kernel,
    merton_path_kernel,
    sabr_jump_path_kernel,
    sabr_path_kernel,
)

__doc__ = """
Defines the American Monte Carlo pricing technique using Longstaff-Schwartz.
"""


class AmericanMonteCarloTechnique(BaseTechnique, GreekMixin, IVMixin):
    """
    Prices American options by simulating full SDE paths and applying the
    Longstaff-Schwartz algorithm.
    """

    def __init__(
        self,
        *,
        n_paths: int = 20_000,
        n_steps: int = 100,
        antithetic: bool = True,
        seed: int | None = None,
        lsm_degree: int = 2,
    ):
        if antithetic and n_paths % 2 != 0:
            n_paths += 1
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.antithetic = antithetic
        self.rng = np.random.default_rng(seed)
        self.lsm_degree = lsm_degree

    def price(
        self,
        option: Option,
        stock: Stock,
        model: BaseModel,
        rate: Rate,
        **kwargs: Any,
    ) -> PricingResult:
        if not model.supports_sde:
            raise TypeError("American MC pricing requires an SDE model.")

        spot_paths = self._simulate_full_sde_paths(option, stock, model, rate, **kwargs)

        price_val = longstaff_schwartz_pricer(
            stock_paths=spot_paths,
            K=option.strike,
            dt=option.maturity / self.n_steps,
            r=rate.get_rate(option.maturity),
            is_call=(option.option_type is OptionType.CALL),
            degree=self.lsm_degree,
        )
        return PricingResult(price=price_val)

    def _get_path_kernel_and_params(
        self,
        model: BaseModel,
        r: float,
        q: float,
        dt: float,
        **kwargs: Any,
    ) -> tuple[Callable, dict[str, Any]]:
        """Selects the JIT-compiled path kernel and prepares its parameters."""
        p = model.params
        kernel_params = {"r": r, "q": q, "dt": dt}

        if isinstance(model, SABRModel | SABRJumpModel):
            kernel_params.update(
                {
                    "alpha": p["alpha"],
                    "beta": p["beta"],
                    "rho": p["rho"],
                    "v0": kwargs.get("v0", p.get("alpha")),
                }
            )
            if model.has_jumps:
                kernel_params.update(
                    {"lambda_": p["lambda"], "mu_j": p["mu_j"], "sigma_j": p["sigma_j"]}
                )
                return sabr_jump_path_kernel, kernel_params
            return sabr_path_kernel, kernel_params
        if model.has_variance_process and model.has_jumps:
            kernel_params.update(
                {
                    "kappa": p["kappa"],
                    "theta": p["theta"],
                    "rho": p["rho"],
                    "vol_of_vol": p["vol_of_vol"],
                    "v0": kwargs.get("v0", p.get("v0")),
                }
            )
            kernel_params.update(
                {"lambda_": p["lambda"], "mu_j": p["mu_j"], "sigma_j": p["sigma_j"]}
            )
            return bates_path_kernel, kernel_params
        elif model.has_variance_process:
            kernel_params.update(
                {
                    "kappa": p["kappa"],
                    "theta": p["theta"],
                    "rho": p["rho"],
                    "vol_of_vol": p["vol_of_vol"],
                    "v0": kwargs.get("v0", p.get("v0")),
                }
            )
            return heston_path_kernel, kernel_params
        elif model.name == "Kou Double-Exponential Jump":
            kernel_params.update(
                {
                    "sigma": p["sigma"],
                    "lambda_": p["lambda"],
                    "p_up": p["p_up"],
                    "eta1": p["eta1"],
                    "eta2": p["eta2"],
                }
            )
            return kou_path_kernel, kernel_params
        elif model.has_jumps:
            kernel_params.update(
                {
                    "sigma": p["sigma"],
                    "lambda_": p["lambda"],
                    "mu_j": p["mu_j"],
                    "sigma_j": p["sigma_j"],
                }
            )
            return merton_path_kernel, kernel_params
        else:
            kernel_params["sigma"] = p.get("sigma")
            return bsm_path_kernel, kernel_params

    def _simulate_full_sde_paths(
        self,
        option: Option,
        stock: Stock,
        model: BaseModel,
        rate: Rate,
        **kwargs: Any,
    ) -> np.ndarray:
        """Simulates and returns the full matrix of stock price paths."""
        S0, T = stock.spot, option.maturity
        r, q = rate.get_rate(T), stock.dividend
        dt = T / self.n_steps
        num_draws = self.n_paths // 2 if self.antithetic else self.n_paths

        kernel, kernel_params = self._get_path_kernel_and_params(
            model, r, q, dt, **kwargs
        )

        sim_params = {
            "n_paths": num_draws,
            "n_steps": self.n_steps,
            **kernel_params,
        }
        if isinstance(model, SABRModel | SABRJumpModel):
            sim_params["s0"] = S0
        else:
            sim_params["log_s0"] = math.log(S0)

        # Generate standard normal draws (z) first, then scale by sqrt(dt)
        if model.has_variance_process:
            z1 = self.rng.standard_normal(size=(num_draws, self.n_steps))
            z2 = self.rng.standard_normal(size=(num_draws, self.n_steps))
            sim_params["dw1"] = z1 * math.sqrt(dt)
            sim_params["dw2"] = z2 * math.sqrt(dt)
        else:
            z = self.rng.standard_normal(size=(num_draws, self.n_steps))
            sim_params["dw"] = z * math.sqrt(dt)

        if model.has_jumps:
            sim_params["jump_counts"] = self.rng.poisson(
                lam=model.params["lambda"] * dt, size=(num_draws, self.n_steps)
            )

        paths = kernel(**sim_params)

        if self.antithetic:
            if "dw" in sim_params:
                sim_params["dw"] *= -1
            if "dw1" in sim_params:
                sim_params["dw1"] *= -1
                sim_params["dw2"] *= -1

            paths_anti = kernel(**sim_params)
            paths = np.concatenate([paths, paths_anti])

        return paths
