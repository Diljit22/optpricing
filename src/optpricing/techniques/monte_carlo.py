from __future__ import annotations

import math
import weakref
from collections.abc import Callable
from typing import Any

import numba
import numpy as np

from optpricing.atoms import Option, OptionType, Rate, Stock
from optpricing.models import BaseModel
from optpricing.techniques.base import BaseTechnique, GreekMixin, IVMixin, PricingResult
from optpricing.techniques.base.random_utils import (
    DEFAULT_CHUNK_SIZE,
    plan_chunks,
    run_chunks,
)

from .kernels.mc_kernels import (
    bates_kernel,
    bates_qe_kernel,
    bsm_kernel,
    dupire_kernel,
    heston_kernel,
    heston_qe_kernel,
    kou_kernel,
    merton_kernel,
    sabr_jump_kernel,
    sabr_kernel,
)

__doc__ = """
Defines a universal Monte Carlo pricing engine that dispatches to specialized,
JIT-compiled kernels based on the financial model provided.
"""

_jitted_surfaces: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _jit_vol_surface(func: Callable) -> Callable:
    """The local-volatility surface ``sigma(t, S)`` as a numba function, compiled
    once per Python callable."""
    if numba.config.DISABLE_JIT or isinstance(func, numba.core.registry.CPUDispatcher):
        return func
    jitted = _jitted_surfaces.get(func)
    if jitted is None:
        jitted = numba.njit(func)
        try:
            jitted.compile("float64(float64, float64)")
        except numba.core.errors.NumbaError as e:
            raise TypeError(
                "The Dupire 'vol_surface' must be a scalar function sigma(t, S) "
                f"that numba can compile in nopython mode: {e}"
            ) from None
        _jitted_surfaces[func] = jitted
    return jitted


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
        n_jobs: int | None = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        variance_scheme: str = "qe",
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
        n_jobs : int | None, optional
            Maximum number of threads simulating path chunks concurrently. None
            uses ``OPTPRICING_MC_THREADS`` or the CPU count. Results do not
            depend on this value.
        chunk_size : int, optional
            Paths per chunk (and per random stream), by default 8192. Results
            depend on the seed and on this value only.
        variance_scheme : {"qe", "euler"}, optional
            Discretisation of the Heston/Bates variance process: Andersen's
            Quadratic-Exponential scheme with martingale correction (default,
            accurate with ~50 steps even when the Feller condition fails) or
            full-truncation Euler.
        """
        if antithetic and n_paths % 2 != 0:
            n_paths += 1
        if variance_scheme not in ("qe", "euler"):
            raise ValueError("variance_scheme must be 'qe' or 'euler'.")
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.antithetic = antithetic
        self.rng = np.random.default_rng(seed)
        self.n_jobs = n_jobs
        self.chunk_size = int(chunk_size)
        self.variance_scheme = variance_scheme

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
            ST = model.sample_terminal_spot(S0, r, T, self.n_paths, q=q, rng=self.rng)
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

        if getattr(model, "is_sabr", False):
            # alpha = initial volatility sigma_0 (overridable via the legacy
            # `v0` kwarg), nu = volatility of volatility. The forward is set
            # by the caller (`f0`).
            kernel_params = {
                "v0": kwargs.get("v0", p["alpha"]),
                "nu": p["nu"],
                "beta": p["beta"],
                "rho": p["rho"],
                "dt": dt,
            }
            if model.has_jumps:
                kernel_params.update(
                    {"lambda_": p["lambda"], "mu_j": p["mu_j"], "sigma_j": p["sigma_j"]}
                )
                return sabr_jump_kernel, kernel_params
            return sabr_kernel, kernel_params

        kernel_params = {"r": r, "q": q, "dt": dt}

        if getattr(model, "is_local_vol", False):  # Dupire
            kernel_params["vol_surface_func"] = _jit_vol_surface(p["vol_surface"])
            return dupire_kernel, kernel_params

        if model.has_variance_process and model.has_jumps:  # Bates
            kernel_params.update(
                {
                    "v0": kwargs.get("v0", p.get("v0")),
                    "kappa": p["kappa"],
                    "theta": p["theta"],
                    "rho": p["rho"],
                    "vol_of_vol": p["vol_of_vol"],
                    "lambda_": p["lambda"],
                    "mu_j": p["mu_j"],
                    "sigma_j": p["sigma_j"],
                }
            )
            return (
                bates_qe_kernel if self.variance_scheme == "qe" else bates_kernel
            ), kernel_params
        elif model.has_variance_process:  # Heston
            kernel_params.update(
                {
                    "v0": kwargs.get("v0", p.get("v0")),
                    "kappa": p["kappa"],
                    "theta": p["theta"],
                    "rho": p["rho"],
                    "vol_of_vol": p["vol_of_vol"],
                }
            )
            return (
                heston_qe_kernel if self.variance_scheme == "qe" else heston_kernel
            ), kernel_params
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
        """
        Simulates the terminal spot with the dispatched kernel.

        Paths are split into chunks of `chunk_size`; each chunk owns an
        independent generator spawned from one draw of `self.rng` (so common
        random numbers keep working) and the numba kernels run concurrently.
        """
        dt = T / self.n_steps
        kernel, kernel_params = self._get_sde_kernel_and_params(
            model, r, q, dt, **kwargs
        )
        is_sabr = getattr(model, "is_sabr", False)
        if is_sabr:
            # SABR is specified on the (shifted) forward; the kernel returns the
            # terminal forward, which is the terminal spot.
            kernel_params["f0"] = S0 * math.exp((r - q) * T) + model.shift
        else:
            kernel_params["log_s0"] = math.log(S0)

        generators, sizes = plan_chunks(
            self.rng,
            self.n_paths,
            chunk_size=self.chunk_size,
            antithetic=self.antithetic,
        )
        terminal = run_chunks(
            kernel,
            generators,
            sizes,
            n_jobs=self.n_jobs,
            n_steps=self.n_steps,
            antithetic=self.antithetic,
            **kernel_params,
        )
        return terminal - model.shift if is_sabr else np.exp(terminal)

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
