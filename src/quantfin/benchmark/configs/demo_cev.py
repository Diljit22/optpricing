import numpy as np

from quantfin.atoms.option                               import Option, OptionType
from quantfin.atoms.stock                                import Stock
from quantfin.atoms.rate                                 import Rate
from quantfin.models.bsm                                 import BSMModel
from quantfin.models.cev import CEVModel
from quantfin.techniques.closed_form                      import ClosedFormTechnique
from quantfin.techniques.crr                              import CRRLatticeTechnique
from quantfin.techniques.leisen_reimer                    import LeisenReimerTechnique
from quantfin.techniques.pde                              import PDESolverTechnique
from quantfin.techniques.topm                             import TOPMLatticeTechnique
from quantfin.benchmark.configs.base_config               import BenchmarkConfig
from quantfin.techniques.integration                      import IntegrationTechnique
from quantfin.techniques.fft                              import FFTTechnique
from quantfin.techniques.monte_carlo                      import MonteCarloTechnique


cev_params = {"sigma": 0.8, "gamma": 0.7}
model = CEVModel(params=cev_params)

stock = Stock(spot=100.0, dividend=0.01)
rate = Rate(rate=0.03)
options_to_price = [
    Option(strike=100.0, maturity=1.0, option_type=OptionType.CALL),
    Option(strike=100.0, maturity=1.0, option_type=OptionType.PUT),
]

config = BenchmarkConfig(
    name="Constant Elasticity of Variance (CEV) Model",
    model=model,
    model_params=cev_params,
    techniques=[
        # Set n_steps=1 to use the exact terminal sampler
        (MonteCarloTechnique(n_paths=50000, n_steps=1, seed=42), "MC (Exact)")
    ],
    stock=stock,
    rate=rate,
    options=options_to_price,
)