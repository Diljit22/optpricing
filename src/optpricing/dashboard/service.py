from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from optpricing.atoms import Rate, Stock
from optpricing.calibration import VolatilitySurface
from optpricing.calibration.technique_selector import select_fastest_technique
from optpricing.data import get_live_option_chain, load_market_snapshot
from optpricing.workflows import DailyWorkflow

__doc__ = """
Defines the main service layer that orchestrates all backend logic for the
Streamlit dashboard, from data loading to model calibration and plot generation.
"""


class DashboardService:
    def __init__(self, ticker: str, snapshot_date: str, model_configs: dict[str, Any]):
        self.ticker = ticker
        self.snapshot_date = snapshot_date
        self.model_configs = model_configs
        self._market_data: pd.DataFrame | None = None
        self.calibrated_models: dict[str, Any] = {}
        self.summary_df: pd.DataFrame | None = None
        self.market_surface: pd.DataFrame | None = None
        self.model_surfaces: dict[str, pd.DataFrame] = {}
        self.final_rate: Rate | None = None
        self.final_stock: Stock | None = None

    @property
    def market_data(self) -> pd.DataFrame:
        if self._market_data is None:
            if self.snapshot_date == "Live Data":
                self._market_data = get_live_option_chain(self.ticker)
            else:
                self._market_data = load_market_snapshot(
                    self.ticker, self.snapshot_date
                )
            if self._market_data is None or self._market_data.empty:
                raise ValueError(
                    f"Could not load data for {self.ticker} on {self.snapshot_date}"
                )
        return self._market_data

    def run_calibrations(self):
        all_results = []
        for model_name, config in self.model_configs.items():
            with st.spinner(f"Running workflow for {model_name}..."):
                config["ticker"] = self.ticker
                workflow = DailyWorkflow(
                    market_data=self.market_data, model_config=config
                )
                workflow.run()
                all_results.append(workflow.results)

                if workflow.results.get("Status") == "Success":
                    model_class = config["model_class"]
                    calibrated_params = workflow.results.get("Calibrated Params")

                    full_params = model_class.default_params.copy()
                    full_params.update(calibrated_params)
                    self.calibrated_models[model_name] = model_class(params=full_params)

                    if self.final_rate is None and hasattr(workflow, "rate"):
                        self.final_rate = workflow.rate
                        self.final_stock = workflow.stock
                        st.sidebar.info(
                            f"Workflow r: {self.final_rate.get_rate():.2%}, "
                            f"q: {self.final_stock.dividend:.2%}"
                        )

        if not all_results:
            raise RuntimeError("Calibration workflows failed to produce any results.")
        self.summary_df = pd.DataFrame(all_results).set_index("Model")

    def calculate_iv_surfaces(self):
        if not self.final_rate or not self.final_stock:
            msg_ = (
                "Cannot calculate IV surfaces without a"
                + " successful calibration run to determine r and q"
            )
            raise RuntimeError(msg_)

        with st.spinner("Calculating Market IV surface..."):
            self.market_surface = (
                VolatilitySurface(self.market_data)
                .calculate_market_iv(self.final_stock, self.final_rate)
                .surface
            )

        for name, model in self.calibrated_models.items():
            with st.spinner(f"Calculating {name} IV surface..."):
                technique = select_fastest_technique(model)
                self.model_surfaces[name] = (
                    VolatilitySurface(self.market_data)
                    .calculate_model_iv(
                        self.final_stock, self.final_rate, model, technique
                    )
                    .surface
                )
