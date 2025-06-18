from __future__ import annotations
import math
from typing import Any, Callable
import numpy as np

from quantfin.atoms.option                   import Option, OptionType
from quantfin.atoms.rate                     import Rate
from quantfin.atoms.stock                    import Stock
from quantfin.models.base                    import BaseModel
from quantfin.techniques.base.base_technique import BaseTechnique, PricingResult
from quantfin.techniques.base.greek_mixin    import GreekMixin
from quantfin.techniques.base.iv_mixin       import IVMixin


class MonteCarloTechnique(BaseTechnique, GreekMixin, IVMixin):
    """
    Universal Monte Carlo engine with adaptive kernel dispatch.
    - Supports log-normal, stochastic variance, and pure Lévy models.
    - Implements antithetic variates for variance reduction.
    """
    def __init__(
        self,
        *,
        n_paths: int = 20_000,
        n_steps: int = 100,
        antithetic: bool = True,
        seed: int | None = None,
    ) -> None:
        if antithetic and n_paths % 2:
            n_paths += 1
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.antithetic = antithetic
        self.rng = np.random.default_rng(seed)

    def price(self, option: Option, stock: Stock, model: BaseModel, rate: Rate, **kwargs: Any) -> PricingResult:
        if not (model.supports_sde or model.is_pure_levy):
            raise TypeError(f"Model '{model.name}' does not support an SDE.")

        S0, K, T = stock.spot, option.strike, option.maturity
        r, q = rate.rate, stock.dividend

        # --- Dispatch to the best simulation kernel ---
        if model.is_pure_levy:
            ST = self._simulate_levy_terminal(model, S0, r, q, T, **kwargs)
        else: # All other models use step-by-step SDE
            ST = self._simulate_sde_path(model, S0, r, q, T, **kwargs)

        # --- Calculate Payoff and Price ---
        payoff = np.maximum(ST - K, 0.0) if option.option_type is OptionType.CALL else np.maximum(K - ST, 0.0)
        price = float(np.mean(payoff) * math.exp(-r * T))
        return PricingResult(price=price)

    def _simulate_sde_path(self, model: BaseModel, S0: float, r: float, q: float, T: float, **kwargs) -> np.ndarray:
        """Kernel for all models that are simulated step-by-step."""
        dt = T / self.n_steps
        sde_stepper = model.sde()
        num_draws = self.n_paths // 2 if self.antithetic else self.n_paths

        # Generate all necessary random numbers
        dw1 = self.rng.standard_normal(size=(num_draws, self.n_steps)) * math.sqrt(dt)
        dw2 = self.rng.standard_normal(size=(num_draws, self.n_steps)) * math.sqrt(dt) if model.has_variance_process else None
        jump_counts = self.rng.poisson(lam=model.params["lambda"] * dt, size=(num_draws, self.n_steps)) if model.has_jumps else None

        # Run simulations
        log_ST = self._run_path_simulation(sde_stepper, model, S0, r, q, dt, dw1, dw2, jump_counts, **kwargs)
        
        if self.antithetic:
            dw1_anti = -dw1
            dw2_anti = -dw2 if dw2 is not None else None
            # reuse the same jump counts for the antithetic path
            log_ST_anti = self._run_path_simulation(sde_stepper, model, S0, r, q, dt, dw1_anti, dw2_anti, jump_counts, **kwargs)
            log_ST = np.concatenate([log_ST, log_ST_anti])

        return np.exp(log_ST)

    def _run_path_simulation(self, stepper, model, S0, r, q, dt, dw1, dw2, jump_counts, **kwargs):
        """The core loop that drives the SDE forward."""
        log_s_t = np.full(dw1.shape[0], math.log(S0))
        v_t = np.full(dw1.shape[0], kwargs.get("v0", model.params.get("v0"))) if model.has_variance_process else None

        for i in range(self.n_steps):
            t_current = i * dt
            if model.name == "Dupire Local Volatility":
                log_s_t = stepper(log_s_t, r, q, dt, dw1[:, i], t_current)
            elif model.has_variance_process and model.has_jumps: # Bates
                log_s_t, v_t = stepper(log_s_t, v_t, r, q, dt, dw1[:, i], dw2[:, i], jump_counts[:, i], self.rng)
            elif model.has_variance_process: # Heston
                log_s_t, v_t = stepper(log_s_t, v_t, r, q, dt, dw1[:, i], dw2[:, i])
            elif model.has_jumps: # Merton, Kou
                log_s_t = stepper(log_s_t, r, q, dt, dw1[:, i], jump_counts[:, i], self.rng)
            else: # BSM
                log_s_t = stepper(log_s_t, r, q, dt, dw1[:, i])
        
        return log_s_t

    def _simulate_levy_terminal(self, model: BaseModel, S0: float, r: float, q: float, T: float, **kwargs) -> np.ndarray:
        """
        Kernel for pure Lévy models using direct terminal sampling.
        """
        if not hasattr(model, "sample_terminal_log_return") or not hasattr(model, "raw_cf"):
            raise NotImplementedError(f"Model '{model.name}' needs 'sample_terminal_log_return' and 'raw_cf' methods.")
        
        num_draws = self.n_paths
        
        # 1. Draw from the model's raw distribution (mean-zero if symmetric)
        log_returns_raw = model.sample_terminal_log_return(T, num_draws, self.rng)
        
        # 2. Calculate the risk-neutral drift adjustment
        # Get the characteristic function of the raw process (no spot, no drift)
        phi_raw = model.raw_cf(t=T)
        # The compensator is log(E[exp(X_T_raw)]) = log(phi_raw(-i))
        # This should now be a real number.
        compensator_log_moment = np.log(phi_raw(-1j)).real # Take .real to discard tiny imaginary noise
        drift = (r - q) * T - compensator_log_moment
        log_ST = math.log(S0) + drift + log_returns_raw
        
        return np.exp(log_ST)