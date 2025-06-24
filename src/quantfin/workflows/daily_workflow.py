from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from quantfin.atoms import Option, OptionType, Rate, Stock
from quantfin.calibration import (
    Calibrator,
    fit_jump_params_from_history,
    fit_rate_and_dividend,
)
from quantfin.calibration.technique_selector import select_fastest_technique
from quantfin.data import load_historical_returns
from quantfin.models import BaseModel

__doc__ = """
Defines the DailyWorkflow for calibrating and evaluating a single model
on a single day's market data.
"""


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

    def run(self):
        """
        Executes the full calibration and evaluation workflow.

        This method performs all steps in sequence and populates the `self.results`
        dictionary with the outcome, including status, calibrated parameters,
        and final RMSE. It includes error handling to ensure the workflow
        doesn't crash on failure.
        """
        model_name = self.model_config["name"]
        print("\n" + "#" * 60)
        print(f"### Starting Workflow for Model: {model_name}")
        print("#" * 60)

        try:
            # Fit market parameters (r, q)
            spot = self.market_data["spot_price"].iloc[0]
            calls = self.market_data[self.market_data["optionType"] == "call"]
            puts = self.market_data[self.market_data["optionType"] == "put"]
            r, q = fit_rate_and_dividend(calls, puts, spot)
            self.results.update({"Implied Rate": r, "Implied Dividend": q})

            stock = Stock(spot=spot, dividend=q)
            rate = Rate(rate=r)

            # Get initial guess for parameters
            initial_guess = self.model_config["initial_guess"].copy()
            if "historical_params" in self.model_config:
                hist_returns = load_historical_returns(self.model_config["ticker"])
                jump_params = fit_jump_params_from_history(hist_returns)
                initial_guess.update(jump_params)

            # Calibrate the model
            print(f"\n[Step 2] Calibrating {model_name} on front-month options...")
            target_expiry = self.market_data["expiry"].min()
            calibration_slice = self.market_data[
                self.market_data["expiry"] == target_expiry
            ].copy()

            model_instance = self.model_config["model_class"](params=initial_guess)
            calibrator = Calibrator(model_instance, calibration_slice, stock, rate)
            calibrated_params = calibrator.fit(
                initial_guess=initial_guess,
                bounds=self.model_config["bounds"],
                frozen_params=self.model_config.get("frozen", {}),
            )
            self.results["Calibrated Params"] = calibrated_params

            # Evaluate the calibrated model on the full chain
            print(f"\n[Step 3] Evaluating calibrated {model_name} on the full chain...")
            final_model = model_instance.with_params(**calibrated_params)
            rmse = self._evaluate_rmse(final_model, stock, rate)
            self.results["RMSE"] = rmse
            self.results["Status"] = "Success"
            print(f"  -> Evaluation Complete. Final RMSE: {rmse:.4f}")

        except Exception as e:
            print(f"\n!!!!!! WORKFLOW FAILED for {model_name} !!!!!!")
            print(f"Error: {e}")
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
