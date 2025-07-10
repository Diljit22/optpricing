from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from optpricing.atoms import Rate, Stock
from optpricing.calibration import (
    Calibrator,
    fit_jump_params_from_history,
    fit_rate_and_dividend,
)
from optpricing.calibration.vectorized_pricer import price_options_vectorized
from optpricing.data import get_live_dividend_yield, load_historical_returns
from optpricing.models import BaseModel

__doc__ = """
Defines the DailyWorkflow for calibrating and evaluating a single model
on a single day's market data.
"""

logger = logging.getLogger(__name__)


class DailyWorkflow:
    """
    Orchestrates the calibration of a single model for a single snapshot of market data.

    This class encapsulates the entire process for a given day:
    1. Fits market-implied risk-free rate (r) and dividend yield (q).
    2. Prepares initial parameter guesses, optionally using historical data.
    3. Calibrates the model to front-month options.
    4. Evaluates the calibrated model's performance (RMSE) on the full option chain.
    """

    def __init__(
        self,
        market_data: pd.DataFrame,
        model_config: dict[str, Any],
    ):
        """
        Initializes the daily workflow.

        Parameters
        ----------
        market_data : pd.DataFrame
            A DataFrame containing the option chain for a single snapshot date.
        model_config : dict[str, Any]
            A dictionary defining how to calibrate the model.
        """
        self.market_data = market_data
        self.model_config = model_config
        self.results: dict[str, Any] = {"Model": self.model_config["name"]}
        if "ticker" in self.model_config:
            self.results["Ticker"] = self.model_config["ticker"]
        self.stock: Stock | None = None
        self.rate: Rate | None = None

    def _prepare_for_evaluation(self):
        """
        A lightweight setup method for backtesting.

        Prepares the necessary market parameters (r, q) and atoms
        (Stock, Rate) for the evaluation day without running a full calibration.
        """
        ticker = self.results.get("Ticker", "N/A")
        logger.info("Preparing evaluation environment for %s...", ticker)
        spot = self.market_data["spot_price"].iloc[0]
        q = get_live_dividend_yield(ticker)
        calls = self.market_data[self.market_data["optionType"] == "call"]
        puts = self.market_data[self.market_data["optionType"] == "put"]
        r, _ = fit_rate_and_dividend(calls, puts, spot, q_fixed=q)
        self.stock = Stock(spot=spot, dividend=q)
        self.rate = Rate(rate=r)
        logger.info("  -> Eval Day Known q: %.4f, Implied r: %.4f", q, r)

    def run(self):
        """
        Executes the full calibration and evaluation workflow.

        This method performs all steps in sequence and populates the `self.results`
        dictionary with the outcome, including status, calibrated parameters,
        and final RMSE. It includes error handling to ensure the workflow
        doesn't crash on failure.
        """
        model_name = self.model_config["name"]
        ticker = self.results.get("Ticker", "N/A")
        logger.info("=" * 60)
        logger.info(
            "### Starting Workflow for Model: %s on Ticker: %s", model_name, ticker
        )
        logger.info("=" * 60)

        try:
            spot = self.market_data["spot_price"].iloc[0]

            logger.info("[Step 1] Getting live dividend and fitting implied rate...")
            q = get_live_dividend_yield(ticker)
            calls = self.market_data[self.market_data["optionType"] == "call"]
            puts = self.market_data[self.market_data["optionType"] == "put"]
            r, _ = fit_rate_and_dividend(calls, puts, spot, q_fixed=q)

            self.results.update({"Implied Rate": r, "Known Dividend": q})
            self.stock = Stock(spot=spot, dividend=q)
            self.rate = Rate(rate=r)

            logger.info("  -> Known q: %.4f, Implied r: %.4f", q, r)

            logger.info("[Step 2] Filtering market data to liquid options...")
            original_count = len(self.market_data)
            min_moneyness, max_moneyness = 0.85, 1.15

            calibration_data = (
                self.market_data[
                    (self.market_data["strike"] / spot >= min_moneyness)
                    & (self.market_data["strike"] / spot <= max_moneyness)
                ]
                .copy()
                .reset_index(drop=True)
            )
            _data_msg = f"{len(calibration_data)} of {original_count} options"
            logger.info(f"  -> Using {_data_msg} for calibration.")

            logger.info("[Step 3] Preparing dynamic initial guesses...")
            model_class = self.model_config["model_class"]
            if not hasattr(model_class, "default_params"):
                logger.error(f"Model {model_class.name} is missing default_params..")
            model_instance = model_class(params=model_class.default_params)

            if hasattr(model_instance, "param_defs"):
                bounds = {
                    k: (p["min"], p["max"])
                    for k, p in model_instance.param_defs.items()
                }
                initial_guess = {
                    k: p["default"] for k, p in model_instance.param_defs.items()
                }
            else:
                # Fallback if param_defs is not defined
                bounds = self.model_config.get("bounds", {})
                initial_guess = model_instance.params.copy()
            # For any model with 'sigma', use average IV as a smart guess
            if (
                "sigma" in initial_guess
                and "impliedVolatility" in calibration_data.columns
            ):
                avg_iv = calibration_data["impliedVolatility"].mean()
                if pd.notna(avg_iv) and avg_iv > 0.01:
                    initial_guess["sigma"] = avg_iv
                    logger.info(f"  -> Dynamic initial guess for sigma: {avg_iv:.4f}")

            frozen_params_dict = {}
            if model_name == "Merton" and self.model_config.get(
                "use_historical_strategy"
            ):
                logger.info(
                    "  -> Activating Merton strat.: freezing historical jump params..."
                )
                hist_returns = load_historical_returns(ticker)
                jump_params = fit_jump_params_from_history(hist_returns)
                frozen_params_dict.update(jump_params)
                initial_guess.update(jump_params)
                logger.info(f"  -> Historical estimates frozen: {jump_params}")

            # Handle any other frozen parameters defined in the config
            frozen_from_config = self.model_config.get("frozen", {})
            if frozen_from_config:
                frozen_params_dict.update(frozen_from_config)

            # Calibrate the model
            logger.info("[Step 4] Calibrating %s...", model_name)
            calibrator = Calibrator(
                model_instance, calibration_data, self.stock, self.rate
            )
            calibrated_params = calibrator.fit(
                initial_guess=initial_guess,
                bounds=bounds,
                frozen_params=frozen_params_dict,
            )
            self.results["Calibrated Params"] = calibrated_params

            # Evaluate the calibrated model on the full chain
            logger.info(
                "[Step 5] Evaluating calibrated %s on the full chain...", model_name
            )
            final_model = model_instance.with_params(**calibrated_params)
            rmse = self._evaluate_rmse(final_model, self.stock, self.rate)
            self.results["RMSE"] = rmse
            self.results["Status"] = "Success"
            logger.info("  -> Evaluation Complete. Final RMSE: %.4f", rmse)

        except Exception as e:
            logger.error(
                "!!!!!! WORKFLOW FAILED for %s !!!!!!", model_name, exc_info=True
            )
            self.results.update({"RMSE": np.nan, "Status": "Failed", "Error": str(e)})

    def _evaluate_rmse(
        self,
        model: BaseModel,
        stock: Stock,
        rate: Rate,
    ) -> float:
        """Calculates the RMSE of a given model against the full market data."""
        eval_data = self.market_data.reset_index(drop=True)

        model_prices = price_options_vectorized(
            options_df=eval_data,
            stock=stock,
            model=model,
            rate=rate,
        )

        errors = model_prices - eval_data["marketPrice"].to_numpy()
        return np.sqrt(np.mean(np.square(errors)))
