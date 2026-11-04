from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import pandas as pd

from optpricing.atoms import Rate, Stock
from optpricing.calibration import (
    CalibrationSettings,
    Calibrator,
    annotate_chain,
    evaluate_model,
    filter_options,
    fit_jump_params_from_history,
    price_options_vectorized,
)
from optpricing.data import historical_returns_if_available, snapshot_market_params
from optpricing.models import BaseModel

__doc__ = """
Defines the DailyWorkflow for calibrating and evaluating a single model
on a single day's market data.
"""

logger = logging.getLogger(__name__)


class DailyWorkflow:
    """
    Orchestrates the calibration of a single model for a single snapshot of market data.

    The workflow is fully offline:

    1. Reads the market parameters (spot, dividend yield, rate) stored in the
       snapshot (implied from the chain for pre-v2 files).
    2. Selects the calibration set with the settings' liquidity filter and the
       (wider) evaluation set.
    3. Builds the initial guess from the recipe, the ATM implied volatility and,
       for jump models with a local returns history, a historical prior.
    4. Calibrates with :class:`~optpricing.calibration.Calibrator` (IV-space
       weighted objective, multi-start) and evaluates the fit.
    """

    def __init__(
        self,
        market_data: pd.DataFrame,
        model_config: dict[str, Any],
        *,
        settings: CalibrationSettings | None = None,
        allow_mc: bool = False,
        warm_start: dict[str, float] | None = None,
    ):
        """
        Initializes the daily workflow.

        Parameters
        ----------
        market_data : pd.DataFrame
            A DataFrame containing the option chain for a single snapshot date.
        model_config : dict[str, Any]
            A recipe: ``name``, ``model_class`` and optionally ``initial_guess``,
            ``bounds``, ``frozen``, ``historical_prior``, ``requires_mc``,
            ``ticker``.
        settings : CalibrationSettings | None, optional
            Filter / objective / optimiser settings; defaults apply if omitted.
        allow_mc : bool, optional
            Permit Monte Carlo pricing for simulation-only models.
        warm_start : dict[str, float] | None, optional
            A previous calibration used as an extra starting point.
        """
        self.market_data = market_data
        self.model_config = model_config
        self.settings = settings or CalibrationSettings()
        self.allow_mc = allow_mc
        self.warm_start = warm_start
        self.results: dict[str, Any] = {"Model": self.model_config["name"]}
        ticker = self.model_config.get("ticker")
        if ticker is None and "ticker" in getattr(market_data, "columns", []):
            ticker = str(market_data["ticker"].iloc[0])
        if ticker is not None:
            self.results["Ticker"] = ticker
        self.stock: Stock | None = None
        self.rate: Rate | None = None
        self.calibration_data: pd.DataFrame | None = None
        self.evaluation_data: pd.DataFrame | None = None
        self.calibrator: Calibrator | None = None
        self.model: BaseModel | None = None

    # -- set-up ----------------------------------------------------------------
    def _prepare_for_evaluation(self) -> None:
        """
        Sets ``stock``/``rate`` from the snapshot and the evaluation set, without
        calibrating (used by the backtest on the out-of-sample day).
        """
        self.stock, self.rate = snapshot_market_params(self.market_data)
        annotated = annotate_chain(self.market_data, self.stock, self.rate)
        self.evaluation_data = filter_options(
            annotated, self.stock, self.rate, self.settings.liquidity.evaluation()
        )
        logger.info(
            "  -> Market params: spot %.2f, q %.4f, r %.4f; %d options for evaluation",
            self.stock.spot,
            self.stock.dividend,
            self.rate.get_rate(),
            len(self.evaluation_data),
        )

    def _initial_guess_and_bounds(
        self, calibration_data: pd.DataFrame
    ) -> tuple[dict[str, float], dict[str, tuple], dict[str, float]]:
        """Recipe first, ``param_defs`` as fallback; dynamic sigma and jump prior."""
        model_class = self.model_config["model_class"]
        defaults = dict(getattr(model_class, "default_params", {}))
        param_defs = getattr(model_class, "param_defs", {}) or {}

        initial_guess = dict(self.model_config.get("initial_guess") or {})
        if not initial_guess:
            initial_guess = {k: p["default"] for k, p in param_defs.items()} or {
                k: v for k, v in defaults.items() if isinstance(v, int | float)
            }
        bounds = dict(self.model_config.get("bounds") or {})
        for k in initial_guess:
            if k not in bounds and k in param_defs:
                bounds[k] = (param_defs[k]["min"], param_defs[k]["max"])

        # ATM implied volatility is the natural start for any diffusion sigma.
        if "sigma" in initial_guess and "iv_market" in calibration_data.columns:
            atm = calibration_data.iloc[
                (calibration_data["moneyness"] - 1.0).abs().argsort()[:10]
            ]
            atm_iv = float(np.nanmedian(atm["iv_market"]))
            if np.isfinite(atm_iv) and atm_iv > 0.01:
                sigma0 = atm_iv
                if "gamma" in initial_guess and self.stock is not None:
                    # CEV: local vol at the spot is sigma * S^(gamma - 1).
                    sigma0 = atm_iv * self.stock.spot ** (1.0 - initial_guess["gamma"])
                lo, hi = bounds.get("sigma", (0.01, 5.0))
                initial_guess["sigma"] = float(np.clip(sigma0, lo, hi))
                logger.info("  -> Dynamic initial guess for sigma: %.4f", sigma0)
        if "v0" in initial_guess and "iv_market" in calibration_data.columns:
            atm = calibration_data.iloc[
                (calibration_data["moneyness"] - 1.0).abs().argsort()[:10]
            ]
            atm_var = float(np.nanmedian(atm["iv_market"])) ** 2
            if np.isfinite(atm_var):
                lo, hi = bounds.get("v0", (1e-3, 1.0))
                initial_guess["v0"] = float(np.clip(atm_var, lo, hi))
                if "theta" in initial_guess:
                    lo_t, hi_t = bounds.get("theta", (1e-3, 1.0))
                    initial_guess["theta"] = float(np.clip(atm_var, lo_t, hi_t))

        raw_frozen = self.model_config.get("frozen") or {}
        if isinstance(raw_frozen, list | tuple):
            # Legacy recipes listed parameter *names*; nothing sensible can be
            # frozen without values, so fit them.
            logger.warning("Ignoring legacy 'frozen' list %s (no values).", raw_frozen)
            raw_frozen = {}
        frozen = dict(raw_frozen)

        if (
            self.settings.historical_prior
            and (
                self.model_config.get("historical_prior")
                or self.model_config.get("use_historical_strategy")
            )
            and "Ticker" in self.results
        ):
            returns = historical_returns_if_available(self.results["Ticker"])
            if returns is not None and len(returns) > 250:
                jump = fit_jump_params_from_history(returns)
                for key in ("lambda", "mu_j", "sigma_j"):
                    if key in initial_guess and key not in frozen:
                        lo, hi = bounds.get(key, (-np.inf, np.inf))
                        initial_guess[key] = float(np.clip(jump[key], lo, hi))
                logger.info("  -> Historical jump prior applied: %s", jump)
            else:
                logger.info("  -> No local returns history; skipping jump prior.")

        return initial_guess, bounds, frozen

    # -- run -------------------------------------------------------------------
    def run(self) -> None:
        """
        Executes the full calibration and evaluation workflow.

        Populates ``self.results`` with the status, the calibrated parameters and
        the fit metrics; never raises (failures are recorded with ``Status`` =
        ``"Failed"`` and an ``Error`` message).
        """
        model_name = self.model_config["name"]
        ticker = self.results.get("Ticker", "N/A")
        logger.info("### Workflow for %s on %s", model_name, ticker)
        t0 = time.perf_counter()
        try:
            self._prepare_for_evaluation()
            self.results.update(
                {
                    "Implied Rate": self.rate.get_rate(),
                    "Known Dividend": self.stock.dividend,
                }
            )

            annotated = annotate_chain(self.market_data, self.stock, self.rate)
            self.calibration_data = filter_options(
                annotated, self.stock, self.rate, self.settings.liquidity
            )
            logger.info(
                "  -> %d of %d options selected for calibration.",
                len(self.calibration_data),
                len(self.market_data),
            )
            if len(self.calibration_data) < 5:
                raise ValueError(
                    f"Only {len(self.calibration_data)} liquid options after "
                    "filtering; cannot calibrate."
                )

            model_class = self.model_config["model_class"]
            if self.model_config.get("requires_mc") and not self.allow_mc:
                raise NotImplementedError(
                    f"{model_name} needs Monte Carlo pricing; rerun with allow_mc."
                )
            model_instance = model_class(params=dict(model_class.default_params))

            initial_guess, bounds, frozen = self._initial_guess_and_bounds(
                self.calibration_data
            )
            logger.info(
                "  -> Fitting %s (frozen: %s)",
                [p for p in initial_guess if p not in frozen],
                list(frozen),
            )

            self.calibrator = Calibrator(
                model_instance,
                self.calibration_data,
                self.stock,
                self.rate,
                objective=self.settings.objective,
                weighting=self.settings.weighting,
                allow_mc=self.allow_mc,
                feller_penalty=self.settings.feller_penalty,
            )
            calibrated = self.calibrator.fit(
                initial_guess=initial_guess,
                bounds=bounds,
                frozen_params=frozen,
                n_starts=self.settings.n_starts,
                warm_start=self.warm_start if self.settings.warm_start else None,
                time_budget=self.settings.time_budget,
                log_scale=self.model_config.get("log_scale", ()),
            )
            self.results["Calibrated Params"] = calibrated
            fit = self.calibrator.result
            self.results["Calibration IV RMSE"] = fit.rmse_iv
            self.results["Calibration RMSE"] = fit.rmse_price
            self.results["N Calibration Options"] = fit.n_options
            self.results["Evaluations"] = fit.n_evaluations

            self.model = model_instance.with_params(**calibrated)
            summary, buckets, _ = self.evaluate(self.model)
            self.results["RMSE"] = summary["rmse_price"]
            self.results["IV RMSE"] = summary["rmse_iv"]
            self.results["Relative RMSE"] = summary["rmse_rel_price"]
            self.results["N Options"] = summary["n_options"]
            self.results["Buckets"] = buckets
            self.results["Status"] = "Success"
            logger.info(
                "  -> %s: calibration IV RMSE %.4f, evaluation IV RMSE %.4f, "
                "price RMSE %.4f",
                model_name,
                fit.rmse_iv,
                summary["rmse_iv"],
                summary["rmse_price"],
            )
        except Exception as e:
            logger.error("Workflow failed for %s: %s", model_name, e)
            logger.debug("Traceback:", exc_info=True)
            self.results.update({"RMSE": np.nan, "Status": "Failed", "Error": str(e)})
        finally:
            self.results["Runtime (s)"] = time.perf_counter() - t0

    # -- evaluation ------------------------------------------------------------
    def evaluate(
        self,
        model: BaseModel,
        stock: Stock | None = None,
        rate: Rate | None = None,
        data: pd.DataFrame | None = None,
    ) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
        """
        Fit metrics of ``model`` on the evaluation set (or ``data``).

        Returns
        -------
        tuple[dict[str, float], pd.DataFrame, pd.DataFrame]
            Summary metrics, bucket table and per-option residuals (see
            :func:`optpricing.calibration.evaluate_model`).
        """
        stock = stock or self.stock
        rate = rate or self.rate
        if stock is None or rate is None:
            self._prepare_for_evaluation()
            stock, rate = self.stock, self.rate
        if data is None:
            if self.evaluation_data is None:
                self._prepare_for_evaluation()
            data = self.evaluation_data
        return evaluate_model(model, data, stock, rate, allow_mc=self.allow_mc)

    def _evaluate_rmse(
        self,
        model: BaseModel,
        stock: Stock,
        rate: Rate,
    ) -> float:
        """Price RMSE of ``model`` against the full snapshot (legacy metric)."""
        eval_data = self.market_data.reset_index(drop=True)
        model_prices = price_options_vectorized(
            options_df=eval_data,
            stock=stock,
            model=model,
            rate=rate,
            allow_mc=self.allow_mc,
        )
        errors = model_prices - eval_data["marketPrice"].to_numpy()
        return float(np.sqrt(np.mean(np.square(errors))))
