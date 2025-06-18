import pandas as pd
import numpy as np

from quantfin.models.base.base_model        import BaseModel
from quantfin.models.bsm                     import BSMModel
from quantfin.atoms.stock                    import Stock
from quantfin.atoms.rate                     import Rate
from quantfin.atoms.option                   import Option, OptionType
from quantfin.techniques.base.base_technique import BaseTechnique
from quantfin.techniques.base.iv_mixin       import IVMixin
from quantfin.techniques.bsm_iv_solver       import BSMIVSolver

class VolatilitySurface:
    def __init__(self, option_data: pd.DataFrame):
        columns_to_keep = ['strike', 'maturity', 'marketPrice', 'optionType', 'expiry']
        self.data = option_data[columns_to_keep].copy()
        self.surface = None
        # Instantiate the fast solver once
        self.iv_solver = BSMIVSolver()

    def _calculate_ivs(self, stock: Stock, rate: Rate, prices_to_invert: pd.Series) -> np.ndarray:
        """Calculates IVs using the fast, vectorized BSM solver."""
        # The solver takes the option data as a DataFrame
        ivs = self.iv_solver.solve(prices_to_invert.values, self.data, stock, rate)
        return ivs

    def calculate_market_iv(self, stock: Stock, rate: Rate) -> 'VolatilitySurface':
        print("Calculating market implied volatility surface...")
        market_ivs = self._calculate_ivs(stock, rate, self.data['marketPrice'])
        self.surface = self.data.copy()
        self.surface['iv'] = market_ivs
        self.surface.dropna(inplace=True)
        self.surface = self.surface[(self.surface['iv'] > 0.01) & (self.surface['iv'] < 2.0)]
        return self

    def calculate_model_iv(self, stock: Stock, rate: Rate, model: BaseModel, technique: BaseTechnique) -> 'VolatilitySurface':
        print(f"Calculating {model.name} implied volatility surface...")
        model_prices = [technique.price(Option(strike=r['strike'], maturity=r['maturity'], option_type=OptionType.CALL if r['optionType'] == 'call' else OptionType.PUT), stock, model, rate, **model.params).price for _, r in self.data.iterrows()]
        model_ivs = self._calculate_ivs(stock, rate, pd.Series(model_prices))
        self.surface = self.data.copy()
        self.surface['iv'] = model_ivs
        self.surface.dropna(inplace=True)
        return self