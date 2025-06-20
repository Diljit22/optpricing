# src/quantfin/dashboard/service.py

import pandas as pd
from typing import Dict, Any

# Use our new, clean data managers
from quantfin.data.market_data_manager import load_market_snapshot, get_live_option_chain
from quantfin.workflows import DailyWorkflow
from quantfin.calibration import VolatilitySurface
from quantfin.atoms import Stock, Rate
from quantfin.calibration import fit_rate_and_dividend
from quantfin.techniques import select_fastest_technique
from quantfin.dashboard.plots import plot_smiles_by_expiry, plot_iv_surface_3d

class DashboardService:
    """Orchestrates all logic for the Streamlit dashboard."""
    def __init__(self, ticker: str, snapshot_date: str, model_configs: Dict[str, Any]):
        self.ticker = ticker
        self.snapshot_date = snapshot_date
        self.model_configs = model_configs
        self._market_data: pd.DataFrame | None = None
        self._stock: Stock | None = None
        self._rate: Rate | None = None
        self.calibrated_models: Dict[str, Any] = {}
        self.summary_df: pd.DataFrame | None = None

    @property
    def market_data(self) -> pd.DataFrame:
        if self._market_data is None:
            if self.snapshot_date == "Live Data":
                self._market_data = get_live_option_chain(self.ticker)
            else:
                self._market_data = load_market_snapshot(self.ticker, self.snapshot_date)
        return self._market_data

    # ... (The rest of the DashboardService remains the same) ...
    @property
    def stock(self) -> Stock:
        if self._stock is None:
            spot = self.market_data['spot_price'].iloc[0]
            _, div = fit_rate_and_dividend(self.market_data[self.market_data['optionType'] == 'call'], self.market_data[self.market_data['optionType'] == 'put'], spot)
            self._stock = Stock(spot=spot, dividend=div)
        return self._stock

    @property
    def rate(self) -> Rate:
        if self._rate is None:
            spot = self.market_data['spot_price'].iloc[0]
            r, _ = fit_rate_and_dividend(self.market_data[self.market_data['optionType'] == 'call'], self.market_data[self.market_data['optionType'] == 'put'], spot)
            self._rate = Rate(rate=r)
        return self._rate

    def run_calibrations(self):
        """Runs the daily workflow for all selected models."""
        all_results = []
        for model_name, config in self.model_configs.items():
            workflow = DailyWorkflow(market_data=self.market_data, model_config=config)
            workflow.run()
            all_results.append(workflow.results)
            if workflow.results.get('Status') == 'Success':
                params = workflow.results.get('Calibrated Params')
                self.calibrated_models[model_name] = config['model_class'](params=params)
        self.summary_df = pd.DataFrame(all_results)

    def get_iv_plots(self):
        """Generates and returns the smile and surface plots."""
        market_surface = VolatilitySurface(self.market_data).calculate_market_iv(self.stock, self.rate).surface
        
        model_surfaces = {}
        for name, model in self.calibrated_models.items():
            technique = select_fastest_technique(model)
            model_surfaces[name] = VolatilitySurface(self.market_data).calculate_model_iv(self.stock, self.rate, model, technique).surface
            
        smile_fig = plot_smiles_by_expiry(market_surface, model_surfaces, self.ticker, self.snapshot_date)
        surface_fig = plot_iv_surface_3d(market_surface, model_surfaces, self.ticker)
        
        return smile_fig, surface_fig