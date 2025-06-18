import numpy as np

from quantfin.atoms.option                               import Option, OptionType
from quantfin.atoms.stock                                import Stock
from quantfin.atoms.rate                                 import Rate
from quantfin.models.bsm                                 import BSMModel
from quantfin.models.merton_jump import MertonJumpModel
from quantfin.techniques.closed_form                      import ClosedFormTechnique
from quantfin.techniques.crr                              import CRRLatticeTechnique
from quantfin.techniques.leisen_reimer                    import LeisenReimerTechnique
from quantfin.techniques.pde                              import PDESolverTechnique
from quantfin.techniques.topm                             import TOPMLatticeTechnique
from quantfin.benchmark.configs.base_config               import BenchmarkConfig
from quantfin.techniques.integration                      import IntegrationTechnique
from quantfin.techniques.fft                              import FFTTechnique
from quantfin.techniques.monte_carlo                      import MonteCarloTechnique

# Define parameters
merton_params = {
    "sigma": 0.2, "lambda": 0.5, "mu_j": -0.1, "sigma_j": 0.15, "max_sum_terms": 100
}
model = MertonJumpModel(params=merton_params)
stock = Stock(spot=100.0, dividend=0.01)
rate = Rate(rate=0.03)
options_to_price = [
    Option(strike=100.0, maturity=1.0, option_type=OptionType.CALL),
    Option(strike=100.0, maturity=1.0, option_type=OptionType.PUT),
]

# Add IntegrationTechnique to the list
techniques_to_run = [
    (ClosedFormTechnique(), "Closed-Form"),
    (IntegrationTechnique(), "Integration"),
     (FFTTechnique(n=14), "FFT"),
     (MonteCarloTechnique(n_paths=20000, n_steps=100, seed=42), "MC")
]

# Create the config object
config = BenchmarkConfig(
    name="Merton Jump-Diffusion Model",
    model=model,
    model_params=merton_params,
    techniques=techniques_to_run,
    stock=stock,
    rate=rate,
    options=options_to_price,
)