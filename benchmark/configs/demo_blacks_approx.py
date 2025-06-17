import numpy as np
from atoms.option import Option, OptionType, ExerciseStyle
from atoms.stock import Stock
from atoms.rate import Rate
from models.blacks_approx import BlacksApproxModel
from techniques.closed_form import ClosedFormTechnique
from .base_config import BenchmarkConfig

sigma = 0.30
rate_val = 0.1
spot = 40.0
dividends = np.array([0.7, 0.7])
div_times = np.array([3/12, 5/12])

model = BlacksApproxModel(params={"sigma": sigma})


stock = Stock(
    spot=spot,
    volatility=sigma,
    discrete_dividends=dividends,
    ex_div_times=div_times
)
rate = Rate(rate=rate_val)

options_to_price = [
    Option(strike=40.0, maturity=0.5, option_type=OptionType.CALL, exercise_style=ExerciseStyle.AMERICAN),
]

config = BenchmarkConfig(
    name="DEMO: Black's Approximation for American Call",
    model=model,
    model_params={"sigma": sigma},
    techniques=[(ClosedFormTechnique(), "Black's Approx")],
    stock=stock,
    rate=rate,
    options=options_to_price,
    # We only want to see the price for this demo
    metrics_to_run=["Price"],
)