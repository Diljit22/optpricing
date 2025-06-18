import pandas as pd
import numpy as np
from typing import Dict, Any

from quantfin.models.base                        import BaseModel
from quantfin.atoms.stock                        import Stock
from quantfin.atoms.rate                         import Rate
from quantfin.atoms.option                       import Option, OptionType
from quantfin.calibration.calibrator             import Calibrator
from quantfin.calibration.fit_market_params      import fit_rate_and_dividend

class DailyWorkflow:
    def __init__(self, market_data: pd.DataFrame, model_config: Dict[str, Any], historical_params: Dict[str, float] = None):
        self.market_data = market_data
        self.model_config = model_config
        self.historical_params = historical_params or {}
        self.results = {}

    def _prepare_run_config(self) -> Dict[str, Any]:
        run_config = {
            "model_class": self.model_config["model_class"],
            "initial_guess": self.model_config["initial_guess"].copy(),
            "bounds": self.model_config.get("bounds", {}).copy()
        }
        frozen_param_names = self.model_config.get("frozen", [])
        frozen_params_dict = {key: run_config["initial_guess"][key] for key in frozen_param_names}
        for param_name in self.model_config.get("historical_params", []):
            if param_name in self.historical_params:
                run_config["initial_guess"][param_name] = self.historical_params[param_name]
                if param_name in frozen_param_names:
                    frozen_params_dict[param_name] = self.historical_params[param_name]
        run_config["frozen"] = frozen_params_dict
        return run_config

    def run(self):
        run_config = self._prepare_run_config()
        model_class = run_config["model_class"]
        initial_guess = run_config["initial_guess"]
        frozen_params = run_config.get("frozen", {})
        bounds = run_config.get("bounds", {})
        model_name = model_class(params=initial_guess).name
        
        print("\n" + "#"*60)
        print(f"### Starting Workflow for Model: {model_name}")
        print("#"*60)

        spot_price = self.market_data['spot_price'].iloc[0]
        
        print("[Step 1] Fitting r and q from ATM options...")
        calls = self.market_data[self.market_data['optionType'] == 'call']
        puts = self.market_data[self.market_data['optionType'] == 'put']
        implied_r, implied_q = fit_rate_and_dividend(calls, puts, spot_price)
        
        stock = Stock(spot=spot_price, dividend=implied_q)
        rate = Rate(rate=implied_r)
        
        try:
            print(f"\n[Step 2] Calibrating {model_name} on front-month options...")
            target_expiry = self.market_data['expiry'].min()
            calibration_slice = self.market_data[self.market_data['expiry'] == target_expiry].copy()
            
            model_to_calibrate = model_class(params=initial_guess)
            calibrator = Calibrator(model_to_calibrate, calibration_slice, stock, rate)
            calibrated_params = calibrator.fit(initial_guess, bounds, frozen_params)
            
            print(f"\n[Step 3] Evaluating calibrated {model_name} on the full chain...")
            calibrated_model = model_class(params=calibrated_params)
            errors = []
            for _, row in self.market_data.iterrows():
                option = Option(strike=row['strike'], maturity=row['maturity'], option_type=OptionType.CALL if row['optionType'] == 'call' else OptionType.PUT)
                model_price = calibrator.technique.price(option, stock, calibrated_model, rate, **calibrated_params).price
                market_price = row['marketPrice']
                errors.append(model_price - market_price)
            
            rmse = np.sqrt(np.mean(np.square(errors)))
            
            self.results = {
                'Model': model_name, 'RMSE': rmse, 'Implied Rate': implied_r,
                'Implied Dividend': implied_q, 'Calibrated Params': calibrated_params,
                'Status': 'Success'
            }
            print(f"  -> Evaluation Complete. Final RMSE: {self.results['RMSE']:.4f}")

        except Exception as e:
            print(f"\n!!!!!! WORKFLOW FAILED for {model_name} !!!!!!")
            print(f"Error: {e}")
            self.results = {
                'Model': model_name, 'RMSE': np.nan, 'Implied Rate': implied_r,
                'Implied Dividend': implied_q, 'Calibrated Params': {'error': str(e)},
                'Status': 'Failed'
            }
