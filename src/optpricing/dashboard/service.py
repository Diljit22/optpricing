from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import streamlit as st

from optpricing.atoms import Rate, Stock
from optpricing.calibration import CalibrationSettings, VolatilitySurface
from optpricing.calibration.technique_selector import select_fastest_technique
from optpricing.data import get_live_option_chain, load_market_snapshot
from optpricing.workflows import DailyWorkflow, load_tuned_settings

__doc__ = """
Defines the main service layer that orchestrates all backend logic for the
Streamlit dashboard, from data loading to model calibration and plot generation.
"""

logger = logging.getLogger(__name__)

SUMMARY_COLUMNS = (
    "Status",
    "Calibration IV RMSE",
    "IV RMSE",
    "RMSE",
    "N Calibration Options",
    "N Options",
    "Runtime (s)",
    "Settings",
    "Calibrated Params",
    "Error",
)


class DashboardService:
    """
    Loads a chain (live or snapshot), calibrates the selected models with the
    offline :class:`~optpricing.workflows.DailyWorkflow` and builds the market
    and model implied-volatility surfaces for the plots.

    Parameters
    ----------
    ticker : str
        Ticker symbol.
    snapshot_date : str
        ``"Live Data"`` or a stored snapshot date.
    model_configs : dict[str, Any]
        Recipes keyed by model name.
    settings : CalibrationSettings | None, optional
        Calibration settings for every model (defaults if omitted).
    use_tuned : bool, optional
        Prefer the settings cached by ``optpricing tune`` for this ticker when
        available, by default True.
    allow_mc : bool, optional
        Permit Monte Carlo pricing for simulation-only models.
    """

    def __init__(
        self,
        ticker: str,
        snapshot_date: str,
        model_configs: dict[str, Any],
        *,
        settings: CalibrationSettings | None = None,
        use_tuned: bool = True,
        allow_mc: bool = False,
    ):
        self.ticker = ticker
        self.snapshot_date = snapshot_date
        self.model_configs = model_configs
        self.settings = settings or CalibrationSettings()
        self.use_tuned = use_tuned
        self.allow_mc = allow_mc
        self._market_data: pd.DataFrame | None = None
        self.calibrated_models: dict[str, Any] = {}
        self.workflows: dict[str, DailyWorkflow] = {}
        self.summary_df: pd.DataFrame | None = None
        self.bucket_tables: dict[str, pd.DataFrame] = {}
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

    def settings_for(self, model_name: str) -> CalibrationSettings:
        """Tuned settings for this ticker/model when requested and cached."""
        if self.use_tuned:
            tuned = load_tuned_settings(self.ticker, model_name)
            if tuned is not None:
                return tuned
        return self.settings

    def run_calibrations(self) -> None:
        """Calibrates every configured model and fills the summary table."""
        all_results = []
        for model_name, config in self.model_configs.items():
            with st.spinner(f"Running workflow for {model_name}..."):
                config = dict(config)
                config["ticker"] = self.ticker
                settings = self.settings_for(model_name)
                workflow = DailyWorkflow(
                    market_data=self.market_data,
                    model_config=config,
                    settings=settings,
                    allow_mc=self.allow_mc,
                )
                workflow.run()
                self.workflows[model_name] = workflow
                result = dict(workflow.results)
                result["Model"] = model_name
                result["Settings"] = settings.name
                buckets = result.pop("Buckets", None)
                if isinstance(buckets, pd.DataFrame):
                    self.bucket_tables[model_name] = buckets
                all_results.append(result)

                if result.get("Status") == "Success":
                    model_class = config["model_class"]
                    calibrated_params = result.get("Calibrated Params")
                    full_params = dict(model_class.default_params)
                    full_params.update(calibrated_params)
                    self.calibrated_models[model_name] = model_class(params=full_params)

                    if self.final_rate is None and getattr(workflow, "rate", None):
                        self.final_rate = workflow.rate
                        self.final_stock = workflow.stock

        if not all_results:
            raise RuntimeError("Calibration workflows failed to produce any results.")
        frame = pd.DataFrame(all_results).set_index("Model")
        keep = [c for c in SUMMARY_COLUMNS if c in frame.columns]
        frame = frame[keep]
        if "Calibrated Params" in frame.columns:
            frame["Calibrated Params"] = frame["Calibrated Params"].map(
                lambda p: (
                    ", ".join(f"{k}={v:.4g}" for k, v in p.items())
                    if isinstance(p, dict)
                    else p
                )
            )
        if "IV RMSE" in frame.columns:
            frame = frame.sort_values("IV RMSE", na_position="last")
        self.summary_df = frame

    def calculate_iv_surfaces(self) -> None:
        """Market and per-model Black-Scholes implied-volatility surfaces."""
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
                try:
                    self.model_surfaces[name] = (
                        VolatilitySurface(self.market_data)
                        .calculate_model_iv(
                            self.final_stock,
                            self.final_rate,
                            model,
                            technique,
                            allow_mc=self.allow_mc,
                        )
                        .surface
                    )
                except NotImplementedError as e:
                    logger.warning("No IV surface for %s: %s", name, e)
