from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from optpricing.calibration import CalibrationSettings
from optpricing.config import BACKTEST_LOGS_DIR
from optpricing.data import get_available_snapshot_dates, load_market_snapshot
from optpricing.models import BSMModel

from .daily_workflow import DailyWorkflow

__doc__ = """
Defines the BacktestWorkflow for running a model calibration and evaluation
over a series of historical market data snapshots.
"""

logger = logging.getLogger(__name__)


class BacktestWorkflow:
    """
    Walk-forward backtest of one model on one ticker.

    Snapshot dates are processed in ascending order; the model is calibrated on
    day ``t`` and evaluated, with day ``t+1``'s own market parameters, on the
    liquid options of day ``t+1``. Everything is offline (no network calls).
    Two baselines accompany every period:

    * ``BSM-ATM`` - a flat Black-Scholes volatility equal to day ``t``'s ATM
      implied volatility;
    * ``Stale`` - the model parameters calibrated one snapshot earlier (``t-1``)
      applied on ``t+1``, i.e. what skipping the daily re-calibration costs.
    """

    def __init__(
        self,
        ticker: str,
        model_config: dict,
        *,
        settings: CalibrationSettings | None = None,
        allow_mc: bool = False,
        dates: list[str] | None = None,
        warm_start: bool = True,
    ):
        """
        Initializes the backtest workflow.

        Parameters
        ----------
        ticker : str
            The stock ticker to run the backtest for.
        model_config : dict
            A dictionary "recipe" defining how to calibrate the model.
        settings : CalibrationSettings | None, optional
            Calibration settings (filters, objective, optimiser).
        allow_mc : bool, optional
            Permit Monte Carlo pricing for simulation-only models.
        dates : list[str] | None, optional
            Snapshot dates to use (any order); defaults to every stored date.
        warm_start : bool, optional
            Seed each calibration with the previous period's parameters.
        """
        self.ticker = ticker.upper()
        self.model_config = dict(model_config)
        self.model_config.setdefault("ticker", self.ticker)
        self.settings = settings or CalibrationSettings()
        self.allow_mc = allow_mc
        self.dates = sorted(dates) if dates is not None else None
        self.warm_start = warm_start
        self.results: list[dict[str, Any]] = []
        self.bucket_results: list[dict[str, Any]] = []

    # -- helpers ---------------------------------------------------------------
    @staticmethod
    def _atm_vol(calibration_data: pd.DataFrame | None) -> float | None:
        if calibration_data is None or calibration_data.empty:
            return None
        near = calibration_data.iloc[
            (calibration_data["moneyness"] - 1.0).abs().argsort()[:10]
        ]
        vol = float(np.nanmedian(near["iv_market"]))
        return vol if np.isfinite(vol) and vol > 0 else None

    def run(self) -> None:
        """
        Executes the full backtesting loop.

        For each consecutive pair of snapshot dates, calibrates on the earlier
        day and records out-of-sample metrics on the later day in
        ``self.results`` (one row per period) and ``self.bucket_results``.
        """
        dates = self.dates or get_available_snapshot_dates(self.ticker, ascending=True)
        if len(dates) < 2:
            logger.warning(
                "Backtest for %s requires at least 2 days of data. Skipping.",
                self.model_config["name"],
            )
            return

        model_name = self.model_config["name"]
        previous_params: dict[str, float] | None = None
        for calib_date, eval_date in zip(dates[:-1], dates[1:]):
            assert calib_date < eval_date, "dates must be ascending (no look-ahead)"
            logger.info(
                "--- %s: calibrate on %s, evaluate on %s ---",
                model_name,
                calib_date,
                eval_date,
            )
            calib_data = load_market_snapshot(self.ticker, calib_date)
            eval_data = load_market_snapshot(self.ticker, eval_date)
            if calib_data is None or eval_data is None:
                continue

            t0 = time.perf_counter()
            calib_workflow = DailyWorkflow(
                market_data=calib_data,
                model_config=self.model_config,
                settings=self.settings,
                allow_mc=self.allow_mc,
                warm_start=previous_params if self.warm_start else None,
            )
            calib_workflow.run()
            row: dict[str, Any] = {
                "Calibration Date": calib_date,
                "Eval Date": eval_date,
                "Model": model_name,
                "Status": calib_workflow.results["Status"],
            }
            if calib_workflow.results["Status"] != "Success":
                row["Error"] = calib_workflow.results.get("Error")
                row["Out-of-Sample RMSE"] = np.nan
                row["Out-of-Sample IV RMSE"] = np.nan
                self.results.append(row)
                logger.warning("Calibration failed; period skipped (%s).", row["Error"])
                continue

            params = calib_workflow.results["Calibrated Params"]
            model = calib_workflow.model
            eval_workflow = DailyWorkflow(
                market_data=eval_data,
                model_config=self.model_config,
                settings=self.settings,
                allow_mc=self.allow_mc,
            )
            eval_workflow._prepare_for_evaluation()
            summary, buckets, _ = eval_workflow.evaluate(model)

            atm_vol = self._atm_vol(calib_workflow.calibration_data)
            bsm_atm_iv_rmse = np.nan
            if atm_vol is not None:
                bsm_summary, _, _ = eval_workflow.evaluate(BSMModel({"sigma": atm_vol}))
                bsm_atm_iv_rmse = bsm_summary["rmse_iv"]
            # Stale baseline: the parameters fitted one snapshot earlier (t-1)
            # applied on t+1, i.e. what skipping today's re-calibration costs.
            stale_iv_rmse = np.nan
            if previous_params is not None:
                try:
                    stale_model = model.with_params(**previous_params)
                    stale_summary, _, _ = eval_workflow.evaluate(stale_model)
                    stale_iv_rmse = stale_summary["rmse_iv"]
                except ValueError:
                    pass

            row.update(
                {
                    "In-Sample IV RMSE": calib_workflow.results.get(
                        "Calibration IV RMSE"
                    ),
                    "Out-of-Sample IV RMSE": summary["rmse_iv"],
                    "Out-of-Sample RMSE": summary["rmse_price"],
                    "Out-of-Sample Relative RMSE": summary["rmse_rel_price"],
                    "BSM-ATM IV RMSE": bsm_atm_iv_rmse,
                    "Stale-Params IV RMSE": stale_iv_rmse,
                    "N Options": summary["n_options"],
                    "N Calibration Options": calib_workflow.results.get(
                        "N Calibration Options"
                    ),
                    "Runtime (s)": time.perf_counter() - t0,
                    "Params": {k: float(v) for k, v in params.items()},
                }
            )
            self.results.append(row)
            for _, b in buckets.iterrows():
                self.bucket_results.append(
                    {"Eval Date": eval_date, "Model": model_name, **b.to_dict()}
                )
            logger.info(
                "Out-of-sample %s on %s: IV RMSE %.4f (BSM-ATM %.4f), price RMSE %.4f",
                model_name,
                eval_date,
                summary["rmse_iv"],
                bsm_atm_iv_rmse,
                summary["rmse_price"],
            )
            previous_params = params

    def summary(self) -> dict[str, Any]:
        """
        Aggregates the per-period rows.

        Returns
        -------
        dict[str, Any]
            ``model``, ``ticker``, ``n_periods``, ``n_failed``, means and medians
            of the out-of-sample IV / price RMSE, the BSM-ATM baseline mean and
            the share of periods beating it.
        """
        frame = pd.DataFrame(self.results)
        ok = frame[frame["Status"] == "Success"] if not frame.empty else frame
        out: dict[str, Any] = {
            "model": self.model_config["name"],
            "ticker": self.ticker,
            "n_periods": int(len(frame)),
            "n_failed": int(len(frame) - len(ok)),
        }
        if ok.empty:
            return out
        oos_iv = ok["Out-of-Sample IV RMSE"].astype(float)
        bsm = ok["BSM-ATM IV RMSE"].astype(float)
        out.update(
            {
                "mean_oos_iv_rmse": float(oos_iv.mean()),
                "median_oos_iv_rmse": float(oos_iv.median()),
                "mean_oos_price_rmse": float(
                    ok["Out-of-Sample RMSE"].astype(float).mean()
                ),
                "mean_in_sample_iv_rmse": float(
                    ok["In-Sample IV RMSE"].astype(float).mean()
                ),
                "mean_bsm_atm_iv_rmse": float(bsm.mean()),
                "share_beating_bsm_atm": float(
                    np.mean(oos_iv.to_numpy() < bsm.to_numpy())
                ),
                "mean_runtime_s": float(ok["Runtime (s)"].astype(float).mean()),
            }
        )
        return out

    def save_results(self) -> tuple[Path, Path] | None:
        """
        Saves the per-period table (CSV), the bucket table (CSV) and the summary
        with parameters (JSON) in the backtest-logs directory.

        Returns
        -------
        tuple[Path, Path] | None
            ``(csv_path, json_path)`` or None when there is nothing to save.
        """
        if not self.results:
            logger.info("No backtest results to save.")
            return None

        model_name = self.model_config["name"].replace(" ", "_")
        today_str = pd.Timestamp.now().strftime("%Y-%m-%d")
        base = BACKTEST_LOGS_DIR / f"{self.ticker}_{model_name}_backtest_{today_str}"
        base.parent.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame(self.results)
        if "Params" in df.columns:
            df["Params"] = df["Params"].map(
                lambda p: json.dumps(p) if isinstance(p, dict) else p
            )
        csv_path = base.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        if self.bucket_results:
            pd.DataFrame(self.bucket_results).to_csv(
                base.with_name(base.name + "_buckets.csv"), index=False
            )
        json_path = base.with_suffix(".json")
        payload = {
            "summary": self.summary(),
            "settings": self.settings.to_dict(),
            "periods": self.results,
        }
        with open(json_path, "w") as f:
            json.dump(payload, f, indent=2, default=_json_default)
        logger.info("Backtest log saved to %s", csv_path)
        return csv_path, json_path


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.integer | np.floating):
        value = obj.item()
        return None if isinstance(value, float) and not np.isfinite(value) else value
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    return str(obj)
