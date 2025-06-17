from atoms.option import Option, OptionType, ExerciseStyle
from atoms.stock import Stock
from atoms.rate import Rate
from models.perpetual_put import PerpetualPutModel
from techniques.closed_form import ClosedFormTechnique
from .base_config import BenchmarkConfig
import numpy as np

sigma = 0.20
rate_val = 0.08
spot = 150.0
dividend = 0.005

model = PerpetualPutModel(params={"sigma": sigma, "rate": rate_val})

stock = Stock(spot=spot, volatility=sigma, dividend=dividend)
rate = Rate(rate=rate_val)

options_to_price = [
    Option(strike=100.0, maturity=np.inf, option_type=OptionType.PUT, exercise_style=ExerciseStyle.AMERICAN),
]

config = BenchmarkConfig(
    name="DEMO: Perpetual American Put Model",
    model=model,
    model_params={"sigma": sigma, "rate": rate_val},
    techniques=[(ClosedFormTechnique(), "Perpetual Put CF")],
    stock=stock,
    rate=rate,
    options=options_to_price,
    # We only want to see the price for this demo
    metrics_to_run=["Price"],
)