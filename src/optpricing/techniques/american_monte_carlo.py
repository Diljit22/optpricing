from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numba
import numpy as np

from optpricing.atoms import Option, OptionType, Rate, Stock
from optpricing.models import (
    BaseModel,
    BSMModel,
    KouModel,
    MertonJumpModel,
    SABRJumpModel,
    SABRModel,
)
from optpricing.techniques.base import BaseTechnique, GreekMixin, IVMixin, PricingResult
from optpricing.techniques.base.random_utils import (
    DEFAULT_CHUNK_SIZE,
    plan_chunks,
    run_chunks,
)

from .kernels.american_mc_kernels import longstaff_schwartz_pricer
from .kernels.path_kernels import (
    bates_path_kernel,
    bates_qe_path_kernel,
    bsm_path_kernel,
    heston_path_kernel,
    heston_qe_path_kernel,
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
        n_jobs: int | None = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        variance_scheme: str = "qe",
    ):
        """
        Parameters
        ----------
        n_paths, n_steps, antithetic, seed, variance_scheme
            As for `MonteCarloTechnique`.
        lsm_degree : int, optional
            Polynomial degree of the Longstaff-Schwartz regression, by default 2.
        n_jobs : int | None, optional
            Maximum number of threads simulating path chunks concurrently;
            results do not depend on it.
        chunk_size : int, optional
            Paths per chunk and random stream, by default 8192.
        """
        if antithetic and n_paths % 2 != 0:
            n_paths += 1
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.antithetic = antithetic
        self.rng = np.random.default_rng(seed)
        self.lsm_degree = lsm_degree
        self.n_jobs = n_jobs
        self.chunk_size = int(chunk_size)
        if variance_scheme not in ("qe", "euler"):
            raise ValueError("variance_scheme must be 'qe' or 'euler'.")
        self.variance_scheme = variance_scheme

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
            n_threads=numba.get_num_threads(),
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
                    "v0": kwargs.get("v0", p["alpha"]),
                    "nu": p["nu"],
                    "beta": p["beta"],
                    "rho": p["rho"],
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
                bates_qe_path_kernel
                if self.variance_scheme == "qe"
                else bates_path_kernel
            ), kernel_params
        if model.has_variance_process:
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
                heston_qe_path_kernel
                if self.variance_scheme == "qe"
                else heston_path_kernel
            ), kernel_params
        if isinstance(model, KouModel):
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
        if isinstance(model, MertonJumpModel):
            kernel_params.update(
                {
                    "sigma": p["sigma"],
                    "lambda_": p["lambda"],
                    "mu_j": p["mu_j"],
                    "sigma_j": p["sigma_j"],
                }
            )
            return merton_path_kernel, kernel_params
        if isinstance(model, BSMModel):
            kernel_params["sigma"] = p["sigma"]
            return bsm_path_kernel, kernel_params
        raise TypeError(
            f"American MC has no path simulation for {type(model).__name__}; "
            "supported: BSM, Merton, Kou, Heston, Bates, SABR, SABR-jump."
        )

    def _simulate_full_sde_paths(
        self,
        option: Option,
        stock: Stock,
        model: BaseModel,
        rate: Rate,
        **kwargs: Any,
    ) -> np.ndarray:
        """Simulates the spot paths, time-major ``(n_steps + 1, n_paths)``."""
        S0, T = stock.spot, option.maturity
        r, q = rate.get_rate(T), stock.dividend
        dt = T / self.n_steps

        kernel, kernel_params = self._get_path_kernel_and_params(
            model, r, q, dt, **kwargs
        )
        is_sabr = isinstance(model, SABRModel | SABRJumpModel)
        if is_sabr:
            # SABR evolves the (shifted) forward; the path kernels convert the
            # forward paths back to spot paths for the exercise decision.
            kernel_params["f0"] = S0 * math.exp((r - q) * T) + model.shift
        else:
            kernel_params["log_s0"] = math.log(S0)

        generators, sizes = plan_chunks(
            self.rng,
            self.n_paths,
            chunk_size=self.chunk_size,
            antithetic=self.antithetic,
        )
        paths = run_chunks(
            kernel,
            generators,
            sizes,
            n_jobs=self.n_jobs,
            axis=1,
            n_steps=self.n_steps,
            antithetic=self.antithetic,
            **kernel_params,
        )

        if is_sabr and model.shift:
            # The kernels discount the shifted forward; remove the shift so the
            # exercise decision sees spot prices.
            horizon = self.n_steps * dt
            times = np.arange(self.n_steps + 1) * dt
            paths = (
                paths - (model.shift * np.exp(-(r - q) * (horizon - times)))[:, None]
            )

        return paths
