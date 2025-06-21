import numpy as np

from quantfin.atoms.option import Option, OptionType
from quantfin.atoms.rate import Rate
from quantfin.atoms.stock import Stock
from quantfin.benchmark.configs.base_config import BenchmarkConfig
from quantfin.models.perpetual_put import PerpetualPutModel
from quantfin.techniques.closed_form import ClosedFormTechnique

sigma = 0.20
rate_val = 0.08
spot = 150.0
dividend = 0.005

model = PerpetualPutModel(params={"sigma": sigma, "rate": rate_val})

stock = Stock(spot=spot, volatility=sigma, dividend=dividend)
rate = Rate(rate=rate_val)

options_to_price = [
    Option(strike=100.0, maturity=np.inf, option_type=OptionType.PUT),
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
