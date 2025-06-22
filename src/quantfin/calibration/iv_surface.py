import numpy as np
import pandas as pd

from quantfin.atoms import Option, OptionType, Rate, Stock
from quantfin.models import BaseModel
from quantfin.techniques.base import BaseTechnique

from .vectorized_bsm_iv import BSMIVSolver


class VolatilitySurface:
    """A class to compute and hold market and model-implied volatility surfaces."""

    def __init__(self, option_data: pd.DataFrame):
        # Ensure the necessary columns are present from the start
        required_cols = ["strike", "maturity", "marketPrice", "optionType", "expiry"]
        if not all(col in option_data.columns for col in required_cols):
            msg = (
                "Input option_data is missing one "
                f"of the required columns: {required_cols}"
            )
            raise ValueError(msg)

        self.data = option_data[required_cols].copy()
        self.surface: pd.DataFrame | None = None
        self.iv_solver = BSMIVSolver()

    def _calculate_ivs(
        self, stock: Stock, rate: Rate, prices_to_invert: pd.Series
    ) -> np.ndarray:
        """Calculates IVs using the fast, vectorized BSM solver."""
        ivs = self.iv_solver.solve(prices_to_invert.values, self.data, stock, rate)
        return ivs

    def calculate_market_iv(self, stock: Stock, rate: Rate) -> "VolatilitySurface":
        """Calculates the market IV surface from market prices."""
        print("Calculating market implied volatility surface...")
        market_ivs = self._calculate_ivs(stock, rate, self.data["marketPrice"])
        self.surface = self.data.copy()
        self.surface["iv"] = market_ivs
        self.surface.dropna(inplace=True)
        self.surface = self.surface[
            (self.surface["iv"] > 1e-4) & (self.surface["iv"] < 2.0)
        ]
        return self

    def calculate_model_iv(
        self, stock: Stock, rate: Rate, model: BaseModel, technique: BaseTechnique
    ) -> "VolatilitySurface":
        """Calculates a model's IV surface by pricing all options and inverting."""
        print(f"Calculating {model.name} implied volatility surface...")

        model_prices = np.array(
            [
                technique.price(
                    Option(
                        strike=row.strike,
                        maturity=row.maturity,
                        option_type=OptionType.CALL
                        if row.optionType == "call"
                        else OptionType.PUT,
                    ),
                    stock,
                    model,
                    rate,
                    **model.params,
                ).price
                for _, row in self.data.iterrows()
            ]
        )

        model_ivs = self._calculate_ivs(
            stock, rate, pd.Series(model_prices, index=self.data.index)
        )
        self.surface = self.data.copy()
        self.surface["iv"] = model_ivs
        self.surface.dropna(inplace=True)
        return self
