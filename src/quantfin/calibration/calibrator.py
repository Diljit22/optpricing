from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar

from quantfin.atoms import Option, OptionType, Rate, Stock
from quantfin.models import BaseModel

from .technique_selector import select_fastest_technique


class Calibrator:
    """A generic class for calibrating financial models to market data."""
    def __init__(self, model: BaseModel, market_data: pd.DataFrame, stock: Stock, rate: Rate):
        self.model = model
        self.market_data = market_data
        self.stock = stock
        self.rate = rate
        self.technique = select_fastest_technique(model)
        print(f"Calibrator using '{self.technique.__class__.__name__}' for model '{model.name}'.")

    def _objective_function(self, params_to_fit_values: np.ndarray, params_to_fit_names: List[str], frozen_params: Dict[str, float]) -> float:
        """The objective function to be minimized, calculating total squared error."""
        current_params = {**frozen_params, **dict(zip(params_to_fit_names, params_to_fit_values))}
        print(f"  Trying params: { {k: f'{v:.4f}' for k, v in current_params.items()} }", end="")
        try:
            temp_model = self.model.with_params(**current_params)
        except ValueError as e:
            print(f" -> Invalid params ({e}), returning large error.")
            return 1e12

        # Vectorized pricing for speed
        # assumes the technique supports vectorized pricing.
        # For now, we fall back to iterating, but this is where future optimization lies.
        total_error = 0.0
        for _, row in self.market_data.iterrows():
            option = Option(strike=row['strike'], maturity=row['maturity'], option_type=OptionType.CALL if row['optionType'] == 'call' else OptionType.PUT)
            model_price = self.technique.price(option, self.stock, temp_model, self.rate, **current_params).price
            total_error += (model_price - row['marketPrice'])**2

        print(f"  --> Total Error: {total_error:.4f}")
        return total_error

    def fit(self, initial_guess: Dict[str, float], bounds: Dict[str, tuple], frozen_params: Dict[str, float] = None) -> Dict[str, float]:
        """Performs the calibration using an optimization algorithm."""
        frozen_params = frozen_params or {}
        params_to_fit_names = [p for p in initial_guess if p not in frozen_params]
        print(f"Fitting parameters: {params_to_fit_names}")
        if not params_to_fit_names:
            return frozen_params

        fit_bounds = [bounds.get(p) for p in params_to_fit_names]
        initial_values = [initial_guess[p] for p in params_to_fit_names]

        if len(params_to_fit_names) == 1:
            # scalar minimizer for one parameter
            res = minimize_scalar(lambda x: self._objective_function(np.array([x]), params_to_fit_names, frozen_params), bounds=fit_bounds[0], method='bounded')
            final_params = {**frozen_params, params_to_fit_names[0]: res.x}
            print(f"Scalar optimization finished. Final loss: {res.fun:.6f}")
        else:
            # gradient-based optimizer for multiple parameters
            res = minimize(fun=self._objective_function, x0=initial_values, args=(params_to_fit_names, frozen_params), method='L-BFGS-B', bounds=fit_bounds)
            final_params = {**frozen_params, **dict(zip(params_to_fit_names, res.x))}
            print(f"Multivariate optimization finished. Final loss: {res.fun:.6f}")
        return final_params
