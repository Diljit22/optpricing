import numpy as np
from scipy.optimize import minimize, minimize_scalar, differential_evolution
from typing import List, Dict, Any
import pandas as pd

from quantfin.calibration.technique_selector import select_fastest_technique
from quantfin.models.base                import BaseModel
from quantfin.atoms.option               import Option, OptionType
from quantfin.atoms.stock                import Stock
from quantfin.atoms.rate                 import Rate



class Calibrator:
    def __init__(self, model: BaseModel, market_data: pd.DataFrame, stock: Stock, rate: Rate):
        self.model = model
        self.market_data = market_data
        self.stock = stock
        self.rate = rate
        self.technique = select_fastest_technique(model)
        print(f"Calibrator using '{self.technique.__class__.__name__}' for model '{model.name}'.")

    def _objective_function(self, params_to_fit_values: np.ndarray, params_to_fit_names: List[str], frozen_params: Dict[str, float]) -> float:
        # 1. Reconstruct the full parameter dictionary
        fitted_params = dict(zip(params_to_fit_names, params_to_fit_values))
        current_params = {**frozen_params, **fitted_params}
        
        print(f"  Trying params: { {k: f'{v:.4f}' for k, v in current_params.items()} }", end="")

        try:
            temp_model = self.model.__class__(params=current_params)
        except ValueError as e:
            print(f" -> Invalid params ({e}), returning large error.")
            return 1e12

        # 2. Calculate model prices and sum squared errors
        total_error = 0.0
        for _, row in self.market_data.iterrows():
            option = Option(
                strike=row['strike'], maturity=row['maturity'],
                option_type=OptionType.CALL if row['optionType'] == 'call' else OptionType.PUT
            )
            # Pass all current params as kwargs for techniques that need them (e.g., Heston's v0)
            model_price = self.technique.price(option, self.stock, temp_model, self.rate, **current_params).price
            market_price = row['marketPrice']
            total_error += (model_price - market_price)**2
            
        print(f"  --> Total Error: {total_error:.4f}")
        return total_error

    def fit(self, initial_guess: Dict[str, float], bounds: Dict[str, tuple], frozen_params: Dict[str, float] = None) -> Dict[str, float]:
        frozen_params = frozen_params or {}
        params_to_fit_names = [p for p in initial_guess if p not in frozen_params]
        
        print(f"Fitting parameters: {params_to_fit_names}")

        if not params_to_fit_names:
            print("No parameters to fit. Returning frozen parameters.")
            return frozen_params

        fit_bounds = [bounds.get(p) for p in params_to_fit_names]
        if any(b is None for b in fit_bounds):
            raise ValueError(f"Bounds must be provided for all fitted parameters: {params_to_fit_names}")

        initial_values = [initial_guess[p] for p in params_to_fit_names]

        # --- Dispatch to the best optimizer ---
        if len(params_to_fit_names) == 1:
            scalar_fun = lambda x: self._objective_function(np.array([x]), params_to_fit_names, frozen_params)
            result = minimize_scalar(fun=scalar_fun, bounds=fit_bounds[0], method='bounded')
            final_params = {**frozen_params, params_to_fit_names[0]: result.x}
            print(f"Scalar optimization finished. Final loss: {result.fun:.6f}")
        else:
            result = minimize(
                fun=self._objective_function,
                x0=initial_values,
                args=(params_to_fit_names, frozen_params),
                method='L-BFGS-B',
                bounds=fit_bounds,
                options={'maxiter': 200, 'disp': False}
            )
            final_params = {**frozen_params, **dict(zip(params_to_fit_names, result.x))}
            print(f"Multivariate optimization finished. Final loss: {result.fun:.6f}")

        return final_params