import numpy as np

from quantfin.atoms.option                               import Option, OptionType
from quantfin.atoms.stock                                import Stock
from quantfin.atoms.rate                                 import Rate
from quantfin.models.sabr import SABRModel
from quantfin.techniques.closed_form                      import ClosedFormTechnique
from quantfin.techniques.crr                              import CRRLatticeTechnique
from quantfin.techniques.leisen_reimer                    import LeisenReimerTechnique
from quantfin.techniques.pde                              import PDESolverTechnique
from quantfin.techniques.topm                             import TOPMLatticeTechnique
from quantfin.benchmark.configs.base_config               import BenchmarkConfig
from quantfin.techniques.integration                      import IntegrationTechnique
from quantfin.techniques.fft                              import FFTTechnique
from quantfin.techniques.monte_carlo                      import MonteCarloTechnique

# Define reasonable SABR parameters
# Note: In SABR, 'alpha' is the vol-of-vol, and 'v0' is the initial volatility.
v0 = 0.20
sabr_params = {"alpha": 0.5, "beta": 0.8, "rho": -0.6}
model = SABRModel(params=sabr_params)

# Market and options
stock = Stock(spot=100.0, dividend=0.01)
rate = Rate(rate=0.03)
options_to_price = [
    Option(strike=100.0, maturity=1.0, option_type=OptionType.CALL),
    Option(strike=100.0, maturity=1.0, option_type=OptionType.PUT),
]

# Create the config object
config = BenchmarkConfig(
    name="SABR Model",
    model=model,
    model_params=sabr_params,
    techniques=[
        (MonteCarloTechnique(n_paths=50000, n_steps=100, seed=42), "MC")
    ],
    stock=stock,
    rate=rate,
    options=options_to_price,
    # Pass the initial volatility v0 to the MC technique
    technique_kwargs={"v0": v0}
)