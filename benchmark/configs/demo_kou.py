from atoms.option import Option, OptionType
from atoms.stock import Stock
from atoms.rate import Rate
from models.kou import KouModel
from techniques.integration import IntegrationTechnique
from techniques.fft import FFTTechnique
from .base_config import BenchmarkConfig
from techniques.monte_carlo import MonteCarloTechnique

# Define reasonable Kou parameters
kou_params = {
    "sigma": 0.15, "lambda": 1.0, "p_up": 0.6, "eta1": 10.0, "eta2": 5.0
}
model = KouModel(params=kou_params)

# Market and options
stock = Stock(spot=100.0, dividend=0.01)
rate = Rate(rate=0.03)
options_to_price = [
    Option(strike=100.0, maturity=1.0, option_type=OptionType.CALL),
    Option(strike=100.0, maturity=1.0, option_type=OptionType.PUT),
]

# Create the config object
config = BenchmarkConfig(
    name="Kou Double-Exponential Jump Model",
    model=model,
    model_params=kou_params,
    techniques=[
        (IntegrationTechnique(), "Integration"),
        (FFTTechnique(n=14), "FFT"),
        (MonteCarloTechnique(n_paths=20000, n_steps=100, seed=42), "MC"),
    ],
    stock=stock,
    rate=rate,
    options=options_to_price,
)
