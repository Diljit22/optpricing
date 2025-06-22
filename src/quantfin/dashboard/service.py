from typing import Any

import pandas as pd

from quantfin.atoms import Rate, Stock
from quantfin.calibration import VolatilitySurface, fit_rate_and_dividend
from quantfin.calibration.technique_selector import select_fastest_technique
from quantfin.dashboard.plots import plot_iv_surface_3d, plot_smiles_by_expiry
from quantfin.data.market_data_manager import (
    get_live_option_chain,
    load_market_snapshot,
)
from quantfin.workflows import DailyWorkflow


class DashboardService:
    """
    Orchestrates all logic for the Streamlit dashboard.

    This class acts as a stateful service to manage data loading,
    model calibration, and result generation for a given ticker and date.

    Parameters
    ----------
    ticker : str
        The stock ticker to analyze.
    snapshot_date : str
        The market data snapshot date ('YYYY-MM-DD') or 'Live Data'.
    model_configs : dict[str, Any]
        A dictionary of model configurations to be used for calibration.
    """

    def __init__(
        self,
        ticker: str,
        snapshot_date: str,
        model_configs: dict[str, Any],
    ):
        self.ticker = ticker
        self.snapshot_date = snapshot_date
        self.model_configs = model_configs
        self._market_data: pd.DataFrame | None = None
        self._stock: Stock | None = None
        self._rate: Rate | None = None
        self.calibrated_models: dict[str, Any] = {}
        self.summary_df: pd.DataFrame | None = None

    @property
    def market_data(self) -> pd.DataFrame:
        """
        Lazily loads the market data for the selected ticker and date.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the option chain market data.
        """
        if self._market_data is None:
            if self.snapshot_date == "Live Data":
                self._market_data = get_live_option_chain(self.ticker)

            else:
                self._market_data = load_market_snapshot(
                    self.ticker, self.snapshot_date
                )
        return self._market_data

    @property
    def stock(self) -> Stock:
        """
        Lazily computes and caches the underlying Stock object.

        The dividend yield is implied from put-call parity.

        Returns
        -------
        Stock
            A Stock instance representing the underlying asset.
        """
        if self._stock is None:
            spot = self.market_data["spot_price"].iloc[0]
            _, div = fit_rate_and_dividend(
                self.market_data[self.market_data["optionType"] == "call"],
                self.market_data[self.market_data["optionType"] == "put"],
                spot,
            )
            self._stock = Stock(spot=spot, dividend=div)

        return self._stock

    @property
    def rate(self) -> Rate:
        """
        Lazily computes and caches the risk-free Rate object.

        The rate is implied from put-call parity.

        Returns
        -------
        Rate
            A Rate instance representing the risk-free rate.
        """
        if self._rate is None:
            spot = self.market_data["spot_price"].iloc[0]
            r, _ = fit_rate_and_dividend(
                self.market_data[self.market_data["optionType"] == "call"],
                self.market_data[self.market_data["optionType"] == "put"],
                spot,
            )
            self._rate = Rate(rate=r)

        return self._rate

    def run_calibrations(self):
        """
        Runs the daily calibration workflow for all selected models.

        This method iterates through the model configurations, runs the
        `DailyWorkflow` for each, and stores the calibrated models and
        a summary of the results.
        """
        all_results = []
        for model_name, config in self.model_configs.items():
            workflow = DailyWorkflow(market_data=self.market_data, model_config=config)
            workflow.run()
            all_results.append(workflow.results)
            if workflow.results.get("Status") == "Success":
                params = workflow.results.get("Calibrated Params")
                self.calibrated_models[model_name] = config["model_class"](
                    params=params
                )
        self.summary_df = pd.DataFrame(all_results)

    def get_iv_plots(self):
        """
        Generates and returns the volatility smile and 3D surface plots.

        It first computes the market IV surface, then computes the IV surface
        for each successfully calibrated model, and finally generates the plots.

        Returns
        -------
        tuple[matplotlib.figure.Figure, plotly.graph_objects.Figure]
            A tuple containing the volatility smile figure and the 3D surface figure.
        """
        market_surface = (
            VolatilitySurface(self.market_data)
            .calculate_market_iv(self.stock, self.rate)
            .surface
        )

        model_surfaces = {}
        for name, model in self.calibrated_models.items():
            technique = select_fastest_technique(model)
            model_surfaces[name] = (
                VolatilitySurface(self.market_data)
                .calculate_model_iv(self.stock, self.rate, model, technique)
                .surface
            )

        smile_fig = plot_smiles_by_expiry(
            market_surface,
            model_surfaces,
            self.ticker,
            self.snapshot_date,
        )
        surface_fig = plot_iv_surface_3d(market_surface, model_surfaces, self.ticker)

        return smile_fig, surface_fig
