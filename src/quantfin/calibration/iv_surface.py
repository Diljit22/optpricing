import pandas as pd
import numpy as np

from quantfin.models import BaseModel
from quantfin.atoms import Stock, Rate, Option, OptionType
from quantfin.techniques.base import BaseTechnique
from .vectorized_bsm_iv import BSMIVSolver

class VolatilitySurface:
    """A class to compute and hold market and model-implied volatility surfaces."""
    def __init__(self, option_data: pd.DataFrame):
        self.data = option_data[['strike', 'maturity', 'marketPrice', 'optionType']].copy()
        self.surface: pd.DataFrame | None = None
        self.iv_solver = BSMIVSolver()

    def calculate_market_iv(self, stock: Stock, rate: Rate) -> 'VolatilitySurface':
        """Calculates the market IV surface from market prices."""
        print("Calculating market implied volatility surface...")
        market_ivs = self.iv_solver.solve(self.data['marketPrice'].values, self.data, stock, rate)
        self.surface = self.data.copy()
        self.surface['iv'] = market_ivs
        self.surface.dropna(inplace=True)
        self.surface = self.surface[(self.surface['iv'] > 1e-4) & (self.surface['iv'] < 2.0)]
        return self

    def calculate_model_iv(self, stock: Stock, rate: Rate, model: BaseModel, technique: BaseTechnique) -> 'VolatilitySurface':
        """Calculates a model's IV surface by pricing all options and inverting."""
        print(f"Calculating {model.name} implied volatility surface...")
        
        # Vectorized Pricing: Create one big Option array and price it.
        # This is a placeholder for full vectorization. For now, we iterate.
        # A fully vectorized technique would handle this in one call.
        model_prices = np.array([
            technique.price(
                Option(strike=row.strike, maturity=row.maturity, option_type=OptionType.CALL if row.optionType == 'call' else OptionType.PUT),
                stock, model, rate, **model.params
            ).price
            for _, row in self.data.iterrows()
        ])
        
        model_ivs = self.iv_solver.solve(model_prices, self.data, stock, rate)
        self.surface = self.data.copy()
        self.surface['iv'] = model_ivs
        self.surface.dropna(inplace=True)
        return self