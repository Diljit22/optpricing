
from quantfin.atoms.option import Option, OptionType
from quantfin.atoms.rate import Rate
from quantfin.atoms.stock import Stock
from quantfin.benchmark.configs.base_config import BenchmarkConfig
from quantfin.models.nig import NIGModel
from quantfin.techniques.fft import FFTTechnique
from quantfin.techniques.integration import IntegrationTechnique
from quantfin.techniques.monte_carlo import MonteCarloTechnique

# Define parameters
nig_params = {"alpha": 15.0, "beta": -5.0, "delta": 0.5}
model = NIGModel(params=nig_params)

# Market and options
stock = Stock(spot=100.0, dividend=0.01)
rate = Rate(rate=0.03)
options_to_price = [
    Option(strike=100.0, maturity=1.0, option_type=OptionType.CALL),
    Option(strike=100.0, maturity=1.0, option_type=OptionType.PUT),
]

# Create the config object
config = BenchmarkConfig(
    name="Normal Inverse Gaussian (NIG) Model",
    model=model,
    model_params=nig_params,
    techniques=[(IntegrationTechnique(), "Integration"), (FFTTechnique(n=14), "FFT"),
                    (MonteCarloTechnique(n_paths=50000, n_steps=500, seed=42), "MC"),
],
    stock=stock,
    rate=rate,
    options=options_to_price,
)
