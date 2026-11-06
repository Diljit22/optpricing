from __future__ import annotations

import logging
import time
import warnings
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import qmc

from optpricing.atoms import Rate, Stock
from optpricing.models import BaseModel

from .market_filters import annotate_chain
from .technique_selector import select_fastest_technique
from .vectorized_bsm_iv import BSMIVSolver, bsm_prices_and_vegas
from .vectorized_pricer import ChainLayout, price_chain, supports_vectorized_pricing

__doc__ = """
Defines the main Calibrator class used to fit financial models to market data.
"""

logger = logging.getLogger(__name__)

_FELLER_KEYS = ("kappa", "theta", "vol_of_vol")


@dataclass
class CalibrationResult:
    """
    Outcome of :meth:`Calibrator.fit`.

    Attributes
    ----------
    params : dict[str, float]
        Calibrated (and frozen) parameters.
    loss : float
        Final weighted mean squared residual (IV² or price² units).
    rmse_iv : float
        Unweighted RMSE of the implied-volatility residuals at the optimum
        (NaN when no option admits an implied volatility).
    rmse_price : float
        Unweighted RMSE of the price residuals at the optimum.
    n_options : int
        Size of the calibration set.
    n_evaluations : int
        Objective evaluations (all starts).
    n_starts : int
        Starting points tried.
    runtime : float
        Wall-clock seconds.
    success : bool
        Whether at least one local optimisation converged.
    message : str
        Optimiser message of the best run.
    starts : list[dict[str, Any]]
        Per-start records (``params``, ``loss``, ``kind``).
    """

    params: dict[str, float]
    loss: float
    rmse_iv: float
    rmse_price: float
    n_options: int
    n_evaluations: int
    n_starts: int
    runtime: float
    success: bool
    message: str = ""
    starts: list[dict[str, Any]] = field(default_factory=list)


class Calibrator:
    """
    Fits a model's parameters to an option chain.

    The residuals are taken in Black-Scholes implied-volatility space by
    default (``objective="iv"``), which puts every strike and maturity on the
    same footing; ``objective="price"`` uses price differences. Residuals can
    be weighted by the inverse bid-ask spread (default), the inverse vega
    (price space) or equally. Parameters are optimised in a unit cube mapped
    onto the bounds with L-BFGS-B, from several starting points (initial
    guess, optional warm start and Latin-hypercube samples); invalid parameter
    sets (model ``ValueError``) and optional Feller violations are penalised.
    """

    def __init__(
        self,
        model: BaseModel,
        market_data: pd.DataFrame,
        stock: Stock,
        rate: Rate,
        *,
        objective: str = "iv",
        weighting: str = "spread",
        allow_mc: bool = False,
        feller_penalty: float = 0.0,
        mc_paths: int = 20_000,
        mc_steps: int = 50,
        random_state: int = 0,
    ):
        """
        Initializes the Calibrator.

        Parameters
        ----------
        model : BaseModel
            The financial model to be calibrated (e.g., HestonModel).
        market_data : pd.DataFrame
            Option quotes: 'strike', 'maturity', 'optionType', 'marketPrice'
            and optionally 'bid'/'ask' (for spread weights) and
            'impliedVolatility' (warm start for the IV solver).
        stock : Stock
            The underlying asset's properties.
        rate : Rate
            The risk-free rate structure.
        objective : {"iv", "price"}, optional
            Residual space, by default "iv".
        weighting : {"spread", "vega", "none"}, optional
            Residual weights, by default "spread".
        allow_mc : bool, optional
            Permit Monte Carlo pricing for simulation-only models.
        feller_penalty : float, optional
            Weight of the soft Feller penalty (Heston/Bates), by default 0.
        mc_paths, mc_steps : int, optional
            Monte Carlo settings when ``allow_mc`` is used.
        random_state : int, optional
            Seed for the Latin-hypercube starting points, by default 0.

        Raises
        ------
        NotImplementedError
            If the model cannot be priced on a chain (no closed form, no
            characteristic function and ``allow_mc`` is False).
        """
        if objective not in ("iv", "price"):
            raise ValueError("objective must be 'iv' or 'price'.")
        if weighting not in ("spread", "vega", "none"):
            raise ValueError("weighting must be 'spread', 'vega' or 'none'.")
        if not supports_vectorized_pricing(model, allow_mc=allow_mc):
            raise NotImplementedError(
                f"{model.name} cannot be calibrated: no closed form or "
                "characteristic function (pass allow_mc=True to use Monte Carlo)."
            )
        self.model = model
        self.stock = stock
        self.rate = rate
        self.objective = objective
        self.weighting = weighting
        self.allow_mc = allow_mc
        self.feller_penalty = float(feller_penalty)
        self.mc_kwargs = {"mc_paths": mc_paths, "mc_steps": mc_steps}
        self.random_state = random_state
        self.technique = select_fastest_technique(model)
        self.iv_solver = BSMIVSolver()
        self.market_data = self._prepare(market_data)
        self.result: CalibrationResult | None = None
        self.n_evaluations = 0
        logger.info(
            "Calibrator: %s on %d options, objective=%s, weighting=%s",
            model.name,
            len(self.market_data),
            objective,
            weighting,
        )

    def _prepare(self, market_data: pd.DataFrame) -> pd.DataFrame:
        data = (
            market_data
            if "iv_market" in market_data.columns
            else annotate_chain(market_data, self.stock, self.rate)
        ).reset_index(drop=True)
        if self.objective == "iv":
            data = data[np.isfinite(data["iv_market"])].reset_index(drop=True)
        self._target_price = data["marketPrice"].to_numpy(dtype=float)
        self._target_iv = data["iv_market"].to_numpy(dtype=float)
        self._weights = self._build_weights(data)
        self._invalid_loss = 1e3 if self.objective == "iv" else 1e6
        self.layout = ChainLayout(data)
        return data

    def _build_weights(self, data: pd.DataFrame) -> np.ndarray:
        n = len(data)
        if n == 0:
            return np.ones(0)
        if self.weighting == "spread" and {"bid", "ask"} <= set(data.columns):
            spread = (data["ask"] - data["bid"]).to_numpy(dtype=float)
            w = 1.0 / np.maximum(np.nan_to_num(spread, nan=np.inf), 0.01)
            w = np.where(
                np.isfinite(w) & (w > 0),
                w,
                np.nanmin(w[w > 0]) if (w > 0).any() else 1.0,
            )
        elif self.weighting == "vega" and self.objective == "price":
            iv = np.where(np.isfinite(self._target_iv), self._target_iv, 0.2)
            T = data["maturity"].to_numpy(dtype=float)
            K = data["strike"].to_numpy(dtype=float)
            r = np.asarray([self.rate.get_rate(t) for t in T])
            is_call = data["optionType"].to_numpy() == "call"
            _, vega = bsm_prices_and_vegas(
                iv, self.stock.spot, K, T, r, self.stock.dividend, is_call
            )
            w = 1.0 / np.maximum(vega, 1e-2) ** 2
        else:
            w = np.ones(n)
        return w / np.mean(w)

    # -- objective -----------------------------------------------------------------
    def _prices(self, model: BaseModel) -> np.ndarray:
        # The optimiser probes near-degenerate corners of the parameter space;
        # a fitted model that is one still warns when it is evaluated.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="COS series truncated", category=RuntimeWarning
            )
            return price_chain(
                self.layout,
                self.stock,
                model,
                self.rate,
                allow_mc=self.allow_mc,
                **self.mc_kwargs,
            )

    def _residuals(self, model: BaseModel) -> np.ndarray:
        prices = self._prices(model)
        if self.objective == "price":
            return prices - self._target_price
        iv = self.iv_solver.solve(
            prices,
            self.market_data,
            self.stock,
            self.rate,
            initial_guess=self._target_iv,
        )
        res = iv - self._target_iv
        # A model price outside the no-arbitrage band has no implied volatility:
        # charge a full vol point so the optimiser moves away from it.
        return np.where(np.isfinite(res), res, 1.0)

    def _penalty(self, params: dict[str, float]) -> float:
        if self.feller_penalty > 0 and all(k in params for k in _FELLER_KEYS):
            gap = params["vol_of_vol"] ** 2 - 2.0 * params["kappa"] * params["theta"]
            return self.feller_penalty * max(gap, 0.0) ** 2
        return 0.0

    def loss(self, params: dict[str, float]) -> float:
        """Weighted mean squared residual (plus penalties) for a parameter set."""
        self.n_evaluations += 1
        try:
            model = self.model.with_params(**params)
        except ValueError as e:
            logger.debug("Invalid params %s (%s)", params, e)
            return self._invalid_loss
        try:
            res = self._residuals(model)
        except (ValueError, FloatingPointError, ZeroDivisionError) as e:
            logger.debug("Pricing failed for %s (%s)", params, e)
            return self._invalid_loss
        if not np.all(np.isfinite(res)):
            return self._invalid_loss
        value = float(np.sum(self._weights * res * res) / np.sum(self._weights))
        return value + self._penalty(params)

    def _objective_function(
        self,
        params_to_fit_values: np.ndarray,
        params_to_fit_names: list[str],
        frozen_params: dict[str, float],
    ) -> float:
        """Total loss for a vector of fitted parameters (legacy entry point)."""
        values = np.asarray(params_to_fit_values, dtype=float)
        params = {**frozen_params, **dict(zip(params_to_fit_names, values))}
        return self.loss(params)

    # -- fit -----------------------------------------------------------------------
    def fit(
        self,
        initial_guess: dict[str, float],
        bounds: dict[str, tuple],
        frozen_params: dict[str, float] | None = None,
        *,
        n_starts: int = 1,
        warm_start: dict[str, float] | None = None,
        n_local: int = 3,
        time_budget: float | None = None,
        maxiter: int = 200,
        log_scale: Iterable[str] = (),
    ) -> dict[str, float]:
        """
        Performs the calibration.

        Parameters
        ----------
        initial_guess : dict[str, float]
            Initial guesses for the parameters to be fitted.
        bounds : dict[str, tuple]
            (min, max) bounds per fitted parameter; missing bounds default to
            ±50 % around the guess.
        frozen_params : dict[str, float] | None, optional
            Parameters held constant. Defaults to None.
        n_starts : int, optional
            Total starting points: the guess, the warm start (if given) and
            Latin-hypercube samples filling up to ``n_starts``. Default 1.
        warm_start : dict[str, float] | None, optional
            Previous calibration used as an extra start point.
        n_local : int, optional
            How many of the best-scoring starts get a full local optimisation.
        time_budget : float | None, optional
            Wall-clock seconds; further starts are skipped once exceeded.
        maxiter : int, optional
            L-BFGS-B iteration cap per start, by default 200.
        log_scale : Iterable[str], optional
            Parameters mapped logarithmically onto their (positive) bounds, for
            scales spanning orders of magnitude (e.g. the CEV ``sigma``, whose
            natural size is ``vol * S^(1-gamma)``).

        Returns
        -------
        dict[str, float]
            The full set of calibrated and frozen parameters. Details are in
            ``self.result`` (:class:`CalibrationResult`).
        """
        t0 = time.perf_counter()
        self.n_evaluations = 0
        frozen = dict(frozen_params or {})
        names = [p for p in initial_guess if p not in frozen]
        if not names:
            self.result = CalibrationResult(
                params=frozen,
                loss=self.loss(frozen) if len(self.market_data) else float("nan"),
                rmse_iv=float("nan"),
                rmse_price=float("nan"),
                n_options=len(self.market_data),
                n_evaluations=self.n_evaluations,
                n_starts=0,
                runtime=time.perf_counter() - t0,
                success=True,
                message="nothing to fit",
            )
            return frozen
        if len(self.market_data) == 0:
            raise ValueError("No options to calibrate to.")

        lo, hi = self._bounds_arrays(names, initial_guess, bounds)
        is_log = np.array([p in set(log_scale) for p in names])
        if np.any(is_log & (lo <= 0)):
            raise ValueError("log_scale parameters need strictly positive bounds.")
        log_lo = np.log(np.where(is_log, lo, 1.0))
        log_hi = np.log(np.where(is_log, hi, np.e))

        def to_unit(x: Any) -> np.ndarray:
            x = np.asarray(x, dtype=float)
            linear = (x - lo) / (hi - lo)
            with np.errstate(divide="ignore", invalid="ignore"):
                logarithmic = (np.log(np.where(is_log, x, 1.0)) - log_lo) / (
                    log_hi - log_lo
                )
            return np.where(is_log, logarithmic, linear)

        def from_unit(u: np.ndarray) -> np.ndarray:
            u = np.clip(u, 0.0, 1.0)
            linear = lo + u * (hi - lo)
            logarithmic = np.exp(log_lo + u * (log_hi - log_lo))
            return np.where(is_log, logarithmic, linear)

        def objective(u: np.ndarray) -> float:
            return self.loss({**frozen, **dict(zip(names, from_unit(u)))})

        # -- starting points
        starts: list[tuple[np.ndarray, str]] = [
            (np.clip(to_unit([initial_guess[p] for p in names]), 0, 1), "guess")
        ]
        if warm_start and all(p in warm_start for p in names):
            starts.append(
                (np.clip(to_unit([warm_start[p] for p in names]), 0, 1), "warm")
            )
        n_random = max(n_starts - len(starts), 0)
        if n_random:
            sampler = qmc.LatinHypercube(d=len(names), seed=self.random_state)
            for u in sampler.random(n_random):
                starts.append((u, "lhs"))

        scored = []
        for u, kind in starts:
            scored.append((objective(u), u, kind))
        scored.sort(key=lambda s: s[0])

        # -- local optimisation from the best starts
        records: list[dict[str, Any]] = []
        best_u, best_loss, best_msg, success = scored[0][1], scored[0][0], "", False
        for i, (loss0, u0, kind) in enumerate(scored[: max(n_local, 1)]):
            if time_budget is not None and i > 0:
                if time.perf_counter() - t0 > time_budget:
                    logger.info("Time budget exhausted after %d start(s).", i)
                    break
            res = minimize(
                objective,
                x0=u0,
                method="L-BFGS-B",
                bounds=[(0.0, 1.0)] * len(names),
                options={"maxiter": maxiter},
            )
            records.append(
                {
                    "kind": kind,
                    "start_loss": float(loss0),
                    "loss": float(res.fun),
                    "params": dict(zip(names, from_unit(res.x))),
                    "success": bool(res.success),
                }
            )
            if res.fun < best_loss:
                best_u, best_loss, best_msg = res.x, float(res.fun), str(res.message)
            success = success or bool(res.success)

        final = {**frozen, **dict(zip(names, from_unit(best_u)))}
        rmse_iv, rmse_price = self._final_rmse(final)
        self.result = CalibrationResult(
            params=final,
            loss=best_loss,
            rmse_iv=rmse_iv,
            rmse_price=rmse_price,
            n_options=len(self.market_data),
            n_evaluations=self.n_evaluations,
            n_starts=len(starts),
            runtime=time.perf_counter() - t0,
            success=success,
            message=best_msg,
            starts=records,
        )
        logger.info(
            "%s calibrated: loss=%.3e rmse_iv=%.4f rmse_price=%.4f "
            "(%d evals, %d starts, %.1fs)",
            self.model.name,
            best_loss,
            rmse_iv,
            rmse_price,
            self.n_evaluations,
            len(starts),
            self.result.runtime,
        )
        return final

    def _bounds_arrays(
        self,
        names: list[str],
        initial_guess: dict[str, float],
        bounds: dict[str, tuple],
    ) -> tuple[np.ndarray, np.ndarray]:
        lo, hi = [], []
        for p in names:
            b = bounds.get(p)
            if b is None or b[0] is None or b[1] is None:
                g = float(initial_guess[p])
                b = (
                    (min(0.5 * g, 1.5 * g), max(0.5 * g, 1.5 * g))
                    if g != 0
                    else (-1, 1)
                )
            if not b[1] > b[0]:
                raise ValueError(f"Invalid bounds for '{p}': {b}")
            lo.append(float(b[0]))
            hi.append(float(b[1]))
        return np.asarray(lo), np.asarray(hi)

    def _final_rmse(self, params: dict[str, float]) -> tuple[float, float]:
        try:
            prices = self._prices(self.model.with_params(**params))
        except Exception:  # pragma: no cover - defensive
            return float("nan"), float("nan")
        price_err = prices - self._target_price
        iv = self.iv_solver.solve(
            prices,
            self.market_data,
            self.stock,
            self.rate,
            initial_guess=self._target_iv,
        )
        iv_err = iv - self._target_iv
        iv_err = iv_err[np.isfinite(iv_err)]
        rmse_iv = float(np.sqrt(np.mean(iv_err**2))) if iv_err.size else float("nan")
        return rmse_iv, float(np.sqrt(np.mean(price_err**2)))
