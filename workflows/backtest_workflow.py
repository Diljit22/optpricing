import pandas as pd
import numpy as np
import os
from datetime import date

from calibration.calibrator import Calibrator
from calibration.fit_market_params import fit_rate_and_dividend
from models.base.base_model import BaseModel
from atoms.stock import Stock
from atoms.rate import Rate
from atoms.option import Option, OptionType
from data.data_manager import load_market_snapshot, get_available_snapshot_dates

class BacktestWorkflow:
    def __init__(self, ticker: str, model_config: dict, historical_params: dict):
        self.ticker = ticker.upper()
        self.model_config = model_config
        self.historical_params = historical_params
        self.results = []

    def run(self):
        available_dates = get_available_snapshot_dates(self.ticker)
        if len(available_dates) < 2:
            print(f"Backtest for {self.model_config['model_name']} requires at least 2 days of data. Skipping.")
            return

        for i in range(len(available_dates) - 1):
            cal_date = available_dates[i]
            eval_date = available_dates[i+1]
            
            print(f"\n--- Processing Period for {self.model_config['model_name']}: Calibrate on {cal_date}, Evaluate on {eval_date} ---")

            cal_data = load_market_snapshot(self.ticker, cal_date)
            eval_data = load_market_snapshot(self.ticker, eval_date)
            if cal_data is None or eval_data is None: continue

            cal_spot = cal_data['spot_price'].iloc[0]
            calls = cal_data[cal_data['optionType'] == 'call']
            puts = cal_data[cal_data['optionType'] == 'put']
            implied_r, implied_q = fit_rate_and_dividend(calls, puts, cal_spot)
            
            cal_stock = Stock(spot=cal_spot, dividend=implied_q)
            cal_rate = Rate(rate=implied_r)
            
            run_config = self._prepare_run_config()
            
            model_to_calibrate = run_config["model_class"](params=run_config["initial_guess"])
            calibrator = Calibrator(model_to_calibrate, cal_data, cal_stock, cal_rate)
            calibrated_params = calibrator.fit(
                initial_guess=run_config["initial_guess"],
                bounds=run_config["bounds"],
                frozen_params=run_config.get("frozen", {})
            )
            
            eval_spot = eval_data['spot_price'].iloc[0]
            eval_stock = Stock(spot=eval_spot, dividend=implied_q)
            eval_rate = Rate(rate=implied_r)
            
            final_model = run_config["model_class"](params=calibrated_params)
            out_of_sample_rmse = self._evaluate_performance(eval_data, final_model, eval_stock, eval_rate)
            
            print(f"  -> Out-of-Sample RMSE for {self.model_config['model_name']} on {eval_date}: {out_of_sample_rmse:.4f}")
            self.results.append({'Eval Date': eval_date, 'Model': self.model_config['model_name'], 'Out-of-Sample RMSE': out_of_sample_rmse})

    def _prepare_run_config(self) -> dict:
        """
        Builds the final config for a calibration run by combining the base
        config with historically fitted parameters.
        """
        # Start with a deep copy of the original config
        run_config = {
            "model_class": self.model_config["model_class"],
            "initial_guess": self.model_config["initial_guess"].copy(),
            "bounds": self.model_config["bounds"].copy()
        }

        # Build the frozen_params DICTIONARY from the LIST of frozen names
        frozen_param_names = self.model_config.get("frozen", [])
        frozen_params_dict = {key: run_config["initial_guess"][key] for key in frozen_param_names}

        # Overwrite guess and frozen values with historical estimates if specified
        if "historical_params" in self.model_config:
            for param_name in self.model_config["historical_params"]:
                if param_name in self.historical_params:
                    run_config["initial_guess"][param_name] = self.historical_params[param_name]
                    if param_name in frozen_param_names:
                        frozen_params_dict[param_name] = self.historical_params[param_name]
        
        run_config["frozen"] = frozen_params_dict
        return run_config

    def _evaluate_performance(self, eval_data, calibrated_model, stock, rate) -> float:
        technique = Calibrator(calibrated_model, eval_data, stock, rate).technique
        errors = [(technique.price(Option(strike=r['strike'], maturity=r['maturity'], option_type=OptionType.CALL if r['optionType'] == 'call' else OptionType.PUT), stock, calibrated_model, rate, **calibrated_model.params).price - r['marketPrice']) for _, r in eval_data.iterrows()]
        return np.sqrt(np.mean(np.square(errors)))