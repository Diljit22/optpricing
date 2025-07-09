from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar

from optpricing.atoms import Rate, Stock
from optpricing.models import BaseModel

from .technique_selector import select_fastest_technique
from .vectorized_pricer import price_options_vectorized

__doc__ = """
Defines the main Calibrator class used to fit financial models to market data.
"""


class Calibrator:
    """
    A generic class for calibrating financial models to market data.

    This class orchestrates the process of finding the model parameters that
    minimize the difference between model prices and observed market prices.
    """

    def __init__(
        self,
        model: BaseModel,
        market_data: pd.DataFrame,
        stock: Stock,
        rate: Rate,
    ):
        """
        Initializes the Calibrator.

        Parameters
        ----------
        model : BaseModel
            The financial model to be calibrated (e.g., HestonModel).
        market_data : pd.DataFrame
            A DataFrame containing market prices of options. Must include
            'strike', 'maturity', 'optionType', and 'marketPrice' columns.
        stock : Stock
            The underlying asset's properties.
        rate : Rate
            The risk-free rate structure.
        """
        self.model = model
        self.market_data = market_data
        self.stock = stock
        self.rate = rate
        self.technique = select_fastest_technique(model)
        _class_name = self.technique.__class__.__name__
        print(f"Calibrator using '{_class_name}' for model '{model.name}'.")

    def _objective_function(
        self,
        params_to_fit_values: np.ndarray,
        params_to_fit_names: list[str],
        frozen_params: dict[str, float],
    ) -> float:
        """The objective function to be minimized, calculating total squared error."""
        native_params = np.asarray(params_to_fit_values, dtype=float)
        current_params = {
            **frozen_params,
            **dict(zip(params_to_fit_names, native_params)),
        }
        print(
            f"  Trying params: { {k: f'{v:.4f}' for k, v in current_params.items()} }",
            end="",
        )
        try:
            temp_model = self.model.with_params(**current_params)
        except ValueError as e:
            print(f" -> Invalid params ({e}), returning large error.")
            return 1e12

        model_prices = price_options_vectorized(
            self.market_data, self.stock, temp_model, self.rate
        )

        # Calculate error against market prices
        total_error = np.sum(
            (model_prices - self.market_data["marketPrice"].values) ** 2
        )

        print(f"  --> RMSE: {np.sqrt(total_error / len(self.market_data)):.6f}")
        return total_error

    def fit(
        self,
        initial_guess: dict[str, float],
        bounds: dict[str, tuple],
        frozen_params: dict[str, float] = None,
    ) -> dict[str, float]:
        """
        Performs the calibration using an optimization algorithm.

        This method uses `scipy.optimize.minimize` (or `minimize_scalar` for
        a single parameter) to find the optimal set of parameters that
        minimizes the objective function.

        Parameters
        ----------
        initial_guess : dict[str, float]
            A dictionary of initial guesses for the parameters to be fitted.
        bounds : dict[str, tuple]
            A dictionary mapping parameter names to their (min, max) bounds.
        frozen_params : dict[str, float] | None, optional
            A dictionary of parameters to hold constant during the optimization.
            Defaults to None.

        Returns
        -------
        dict[str, float]
            A dictionary containing the full set of calibrated and frozen parameters.
        """
        frozen_params = frozen_params or {}
        params_to_fit_names = [p for p in initial_guess if p not in frozen_params]
        print(f"Fitting parameters: {params_to_fit_names}")
        if not params_to_fit_names:
            return frozen_params

        fit_bounds = [bounds.get(p) for p in params_to_fit_names]
        initial_values = [initial_guess[p] for p in params_to_fit_names]

        if len(params_to_fit_names) == 1:
            # scalar minimizer for one parameter
            res = minimize_scalar(
                lambda x: self._objective_function(
                    np.array([x]), params_to_fit_names, frozen_params
                ),
                bounds=fit_bounds[0],
                method="bounded",
            )
            final_params = {**frozen_params, params_to_fit_names[0]: res.x}
            print(f"Scalar optimization finished. Final loss: {res.fun:.6f}")
        else:
            # gradient-based optimizer for multiple parameters
            res = minimize(
                fun=self._objective_function,
                x0=initial_values,
                args=(params_to_fit_names, frozen_params),
                method="L-BFGS-B",
                bounds=fit_bounds,
            )
            final_params = {**frozen_params, **dict(zip(params_to_fit_names, res.x))}
            print(f"Multivariate optimization finished. Final loss: {res.fun:.6f}")
        return final_params
