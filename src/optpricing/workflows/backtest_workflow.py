from __future__ import annotations

import logging

import pandas as pd

from optpricing.config import BACKTEST_LOGS_DIR
from optpricing.data import get_available_snapshot_dates, load_market_snapshot

from .daily_workflow import DailyWorkflow

__doc__ = """
Defines the BacktestWorkflow for running a model calibration and evaluation
over a series of historical market data snapshots.
"""

logger = logging.getLogger(__name__)


class BacktestWorkflow:
    """
    Orchestrates a backtest for a single model over multiple historical snapshots.

    This workflow iterates through available historical data, using each day's
    data to calibrate a model and the subsequent day's data to evaluate its
    out-of-sample performance.
    """

    def __init__(self, ticker: str, model_config: dict):
        """
        Initializes the backtest workflow.

        Parameters
        ----------
        ticker : str
            The stock ticker to run the backtest for.
        model_config : dict
            A dictionary "recipe" defining how to calibrate the model.
        """
        self.ticker = ticker.upper()
        self.model_config = model_config
        self.results = []

    def run(self):
        """
        Executes the full backtesting loop.

        It fetches available dates, then for each calibration/evaluation pair,
        it runs a `DailyWorkflow` to calibrate the model and then evaluates
        the out-of-sample RMSE on the next day's data.
        """
        available_dates = get_available_snapshot_dates(self.ticker)
        if len(available_dates) < 2:
            logger.warning(
                "Backtest for %s requires at least 2 days of data. Skipping.",
                self.model_config["name"],
            )
            return

        for i in range(len(available_dates) - 1):
            calib_date, eval_date = available_dates[i], available_dates[i + 1]

            logger.info(
                "--- Processing Period: Calibrate on %s, Evaluate on %s ---",
                calib_date,
                eval_date,
            )

            calib_data = load_market_snapshot(self.ticker, calib_date)
            eval_data = load_market_snapshot(self.ticker, eval_date)
            if calib_data is None or eval_data is None:
                continue

            # Run a daily workflow to get the calibrated model
            calib_workflow = DailyWorkflow(
                market_data=calib_data, model_config=self.model_config
            )
            calib_workflow.run()

            if calib_workflow.results["Status"] != "Success":
                logger.warning(
                    "Calibration failed. Skipping evaluation for this period."
                )
                continue

            calibrated_model = self.model_config["model_class"](
                params=calib_workflow.results["Calibrated Params"]
            )

            eval_workflow = DailyWorkflow(
                market_data=eval_data,
                model_config=self.model_config,
            )
            eval_workflow._prepare_for_evaluation()

            rmse = eval_workflow._evaluate_rmse(
                calibrated_model, eval_workflow.stock, eval_workflow.rate
            )

            logger.info(
                "Out-of-Sample RMSE for %s on %s: %.4f",
                self.model_config["name"],
                eval_date,
                rmse,
            )

            self.results.append(
                {
                    "Eval Date": eval_date,
                    "Model": self.model_config["name"],
                    "Out-of-Sample RMSE": rmse,
                }
            )

    def save_results(self):
        """
        Saves the collected backtest results to a CSV file in the artifacts directory.
        """
        if not self.results:
            logger.info("No backtest results to save.")
            return

        df = pd.DataFrame(self.results)
        today_str = pd.Timestamp.now().strftime("%Y-%m-%d")
        filepath = BACKTEST_LOGS_DIR / f"{self.ticker}_backtest_{today_str}.csv"
        df.to_csv(filepath, index=False)
        logger.info("Detailed backtest log saved to: %s", filepath)
