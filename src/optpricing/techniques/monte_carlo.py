from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np

from optpricing.atoms import Option, OptionType, Rate, Stock
from optpricing.models import BaseModel
from optpricing.techniques.base import BaseTechnique, GreekMixin, IVMixin, PricingResult

from .kernels.mc_kernels import (
    bates_kernel,
    bsm_kernel,
    dupire_kernel,
    heston_kernel,
    kou_kernel,
    merton_kernel,
    sabr_jump_kernel,
    sabr_kernel,
)

__doc__ = """
Defines a universal Monte Carlo pricing engine that dispatches to specialized,
JIT-compiled kernels based on the financial model provided.
"""


class MonteCarloTechnique(BaseTechnique, GreekMixin, IVMixin):
    """
    A universal Monte Carlo engine that dispatches to
    specialized, JIT-compiled kernels for different model types.
    """

    def __init__(
        self,
        *,
        n_paths: int = 20_000,
        n_steps: int = 100,
        antithetic: bool = True,
        seed: int | None = None,
    ):
        """
        Initializes the Monte Carlo engine.

        Parameters
        ----------
        n_paths : int, optional
            The number of simulation paths, by default 20_000.
        n_steps : int, optional
            The number of time steps in each path, by default 100.
        antithetic : bool, optional
            Whether to use antithetic variates for variance reduction, by default True.
        seed : int | None, optional
            Seed for the random number generator for reproducibility, by default None.
        """
        if antithetic and n_paths % 2 != 0:
            n_paths += 1
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.antithetic = antithetic
        self.rng = np.random.default_rng(seed)

    def price(
        self,
        option: Option,
        stock: Stock,
        model: BaseModel,
        rate: Rate,
        **kwargs: Any,
    ) -> PricingResult:
        """
        Prices an option using the appropriate Monte Carlo simulation method.

        This method acts as a dispatcher, selecting the correct simulation
        strategy (SDE path, pure Levy, or exact sampler) based on the
        capabilities of the provided model.

        Parameters
        ----------
        option : Option
            The option contract to be priced.
        stock : Stock
            The underlying asset's properties.
        model : BaseModel
            The financial model to use for the simulation.
        rate : Rate
            The risk-free rate structure.

        Returns
        -------
        PricingResult
            An object containing the calculated price.
        """
        if not (model.supports_sde or getattr(model, "is_pure_levy", False)):
            raise TypeError(f"Model '{model.name}' does not support simulation.")

        S0, K, T = stock.spot, option.strike, option.maturity
        r, q = rate.get_rate(T), stock.dividend

        # Dispatch to the correct simulation method
        if getattr(model, "has_exact_sampler", False):
            ST = model.sample_terminal_spot(S0, r, T, self.n_paths)
        elif getattr(model, "is_pure_levy", False):
            ST = self._simulate_levy_terminal(model, S0, r, q, T)
        else:
            ST = self._simulate_sde_path(model, S0, r, q, T, **kwargs)

        payoff = (
            np.maximum(ST - K, 0)
            if option.option_type is OptionType.CALL
            else np.maximum(K - ST, 0)
        )
        price = float(np.mean(payoff) * math.exp(-r * T))
        return PricingResult(price=price)

    def _get_sde_kernel_and_params(
        self,
        model: BaseModel,
        r: float,
        q: float,
        dt: float,
        **kwargs: Any,
    ) -> tuple[Callable, dict[str, Any]]:
        """Selects the appropriate JIT-compiled kernel and prepares its parameters."""
        p = model.params
        kernel_params = {"r": r, "q": q, "dt": dt}

        if getattr(model, "is_sabr", False):
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
                return sabr_jump_kernel, kernel_params
            return sabr_kernel, kernel_params

        if getattr(model, "is_local_vol", False):  # Dupire
            return dupire_kernel, kernel_params

        if model.has_variance_process and model.has_jumps:  # Bates
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
            return bates_kernel, kernel_params
        elif model.has_variance_process:  # Heston
            kernel_params.update(
                {
                    "kappa": p["kappa"],
                    "theta": p["theta"],
                    "rho": p["rho"],
                    "vol_of_vol": p["vol_of_vol"],
                    "v0": kwargs.get("v0", p.get("v0")),
                }
            )
            return heston_kernel, kernel_params
        elif model.name == "Kou Double-Exponential Jump":  # Kou Model
            kernel_params.update(
                {
                    "sigma": p["sigma"],
                    "lambda_": p["lambda"],
                    "p_up": p["p_up"],
                    "eta1": p["eta1"],
                    "eta2": p["eta2"],
                }
            )
            return kou_kernel, kernel_params
        elif model.has_jumps:  # Merton
            kernel_params.update(
                {
                    "sigma": p["sigma"],
                    "lambda_": p["lambda"],
                    "mu_j": p["mu_j"],
                    "sigma_j": p["sigma_j"],
                }
            )
            return merton_kernel, kernel_params
        else:  # BSM
            kernel_params["sigma"] = p.get("sigma")
            return bsm_kernel, kernel_params

    def _simulate_sde_path(
        self,
        model: BaseModel,
        S0: float,
        r: float,
        q: float,
        T: float,
        **kwargs: Any,
    ) -> np.ndarray:
        """Prepares random numbers and executes the dispatched SDE kernel."""
        dt = T / self.n_steps
        num_draws = self.n_paths // 2 if self.antithetic else self.n_paths

        kernel, kernel_params = self._get_sde_kernel_and_params(
            model, r, q, dt, **kwargs
        )

        sim_params = {"n_paths": num_draws, "n_steps": self.n_steps, **kernel_params}
        if getattr(model, "is_sabr", False):
            sim_params["s0"] = S0
        else:
            sim_params["log_s0"] = math.log(S0)

        if model.has_variance_process:
            dw_v = self.rng.standard_normal(size=(num_draws, self.n_steps)) * math.sqrt(
                dt
            )
            dw_uncorr = self.rng.standard_normal(
                size=(num_draws, self.n_steps)
            ) * math.sqrt(dt)
            rho = model.params.get("rho", 0)
            sim_params["dw1"] = rho * dw_v + math.sqrt(1 - rho**2) * dw_uncorr
            sim_params["dw2"] = dw_v
        else:
            sim_params["dw"] = self.rng.standard_normal(
                size=(num_draws, self.n_steps)
            ) * math.sqrt(dt)

        if model.has_jumps:
            sim_params["jump_counts"] = self.rng.poisson(
                lam=model.params["lambda"] * dt, size=(num_draws, self.n_steps)
            )

        if getattr(model, "is_local_vol", False):
            sim_params["vol_surface_func"] = model.params["vol_surface"]

        ST = kernel(**sim_params)

        if self.antithetic and not getattr(model, "is_local_vol", False):
            if "dw" in sim_params:
                sim_params["dw"] *= -1
            if "dw1" in sim_params:
                sim_params["dw1"] *= -1
            if "dw2" in sim_params:
                sim_params["dw2"] *= -1
            ST_anti = kernel(**sim_params)
            ST = np.concatenate([ST, ST_anti])

        return ST if getattr(model, "is_sabr", False) else np.exp(ST)

    def _simulate_levy_terminal(
        self,
        model: BaseModel,
        S0: float,
        r: float,
        q: float,
        T: float,
    ) -> np.ndarray:
        """Handles pure Levy models by direct sampling of the terminal distribution."""
        if not hasattr(model, "sample_terminal_log_return") or not hasattr(
            model, "raw_cf"
        ):
            raise NotImplementedError(
                f"Levy model '{model.name}' is missing required methods."
            )

        log_returns_raw = model.sample_terminal_log_return(T, self.n_paths, self.rng)
        phi_raw = model.raw_cf(t=T)
        compensator = np.log(np.real(phi_raw(-1j)))
        drift = (r - q) * T - compensator

        return S0 * np.exp(drift + log_returns_raw)
