from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from optpricing.atoms import Option, OptionType, Rate, Stock
from optpricing.calibration import (
    Calibrator,
    fit_jump_params_from_history,
    fit_rate_and_dividend,
)
from optpricing.calibration.technique_selector import select_fastest_technique
from optpricing.data import load_historical_returns
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
        self.stock: Stock | None = None
        self.rate: Rate | None = None

    def run(self):
        """
        Executes the full calibration and evaluation workflow.

        This method performs all steps in sequence and populates the `self.results`
        dictionary with the outcome, including status, calibrated parameters,
        and final RMSE. It includes error handling to ensure the workflow
        doesn't crash on failure.
        """
        model_name = self.model_config["name"]
        logger.info("=" * 60)
        logger.info("### Starting Workflow for Model: %s", model_name)
        logger.info("=" * 60)

        try:
            # Fit market parameters (r, q)
            logger.info("[Step 1] Fitting market parameters (r, q)...")
            spot = self.market_data["spot_price"].iloc[0]
            calls = self.market_data[self.market_data["optionType"] == "call"]
            puts = self.market_data[self.market_data["optionType"] == "put"]
            r, q = fit_rate_and_dividend(calls, puts, spot)
            self.results.update({"Implied Rate": r, "Implied Dividend": q})

            self.stock = Stock(spot=spot, dividend=q)
            self.rate = Rate(rate=r)
            logger.info("  -> Implied r: %.4f, Implied q: %.4f", r, q)

            # Get initial guess for parameters
            initial_guess = self.model_config["initial_guess"].copy()
            if "historical_params" in self.model_config:
                logger.info("[Step 1.5] Fitting historical jump parameters...")
                hist_returns = load_historical_returns(self.model_config["ticker"])
                jump_params = fit_jump_params_from_history(hist_returns)
                initial_guess.update(jump_params)

            # Calibrate the model
            logger.info("[Step 2] Calibrating %s on front-month options...", model_name)
            target_expiry = self.market_data["expiry"].min()
            calibration_slice = self.market_data[
                self.market_data["expiry"] == target_expiry
            ].copy()

            model_instance = self.model_config["model_class"](params=initial_guess)
            calibrator = Calibrator(
                model_instance,
                calibration_slice,
                self.stock,
                self.rate,
            )
            calibrated_params = calibrator.fit(
                initial_guess=initial_guess,
                bounds=self.model_config["bounds"],
                frozen_params=self.model_config.get("frozen", {}),
            )
            self.results["Calibrated Params"] = calibrated_params

            # Evaluate the calibrated model on the full chain
            logger.info(
                "[Step 3] Evaluating calibrated %s on the full chain...",
                model_name,
            )
            final_model = model_instance.with_params(**calibrated_params)
            rmse = self._evaluate_rmse(final_model, self.stock, self.rate)
            self.results["RMSE"] = rmse
            self.results["Status"] = "Success"
            logger.info("  -> Evaluation Complete. Final RMSE: %.4f", rmse)

        except Exception as e:
            logger.error(
                "!!!!!! WORKFLOW FAILED for %s !!!!!!",
                model_name,
                exc_info=True,
            )
            self.results.update({"RMSE": np.nan, "Status": "Failed", "Error": str(e)})

    def _evaluate_rmse(self, model: BaseModel, stock: Stock, rate: Rate) -> float:
        """Calculates the RMSE of a given model against the full market data."""
        technique = select_fastest_technique(model)
        errors = []
        for _, row in self.market_data.iterrows():
            option = Option(
                strike=row["strike"],
                maturity=row["maturity"],
                option_type=OptionType.CALL
                if row["optionType"] == "call"
                else OptionType.PUT,
            )
            model_price = technique.price(
                option, stock, model, rate, **model.params
            ).price
            errors.append(model_price - row["marketPrice"])
        return np.sqrt(np.mean(np.square(errors)))
