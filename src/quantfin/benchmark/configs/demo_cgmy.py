from quantfin.atoms.option import Option, OptionType
from quantfin.atoms.rate import Rate
from quantfin.atoms.stock import Stock
from quantfin.benchmark.configs.base_config import BenchmarkConfig
from quantfin.models.cgmy import CGMYModel
from quantfin.techniques.fft import FFTTechnique
from quantfin.techniques.integration import IntegrationTechnique

cgmy_params = {"C": 0.02, "G": 5.0, "M": 5.0, "Y": 1.2}
model = CGMYModel(params=cgmy_params)

stock = Stock(spot=100.0, dividend=0.01)
rate = Rate(rate=0.03)
options_to_price = [
    Option(strike=100.0, maturity=1.0, option_type=OptionType.CALL),
    Option(strike=100.0, maturity=1.0, option_type=OptionType.PUT),
]

config = BenchmarkConfig(
    name="CGMY Model",
    model=model,
    model_params=cgmy_params,
    techniques=[(IntegrationTechnique(), "Integration"), (FFTTechnique(n=14), "FFT")],
    stock=stock,
    rate=rate,
    options=options_to_price,
)
