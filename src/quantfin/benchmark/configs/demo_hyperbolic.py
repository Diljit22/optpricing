import numpy as np

from quantfin.atoms.option                               import Option, OptionType
from quantfin.atoms.stock                                import Stock
from quantfin.atoms.rate                                 import Rate
from quantfin.models.bsm                                 import BSMModel
from quantfin.models.hyperbolic import HyperbolicModel
from quantfin.techniques.closed_form                      import ClosedFormTechnique
from quantfin.techniques.crr                              import CRRLatticeTechnique
from quantfin.techniques.leisen_reimer                    import LeisenReimerTechnique
from quantfin.techniques.pde                              import PDESolverTechnique
from quantfin.techniques.topm                             import TOPMLatticeTechnique
from quantfin.benchmark.configs.base_config               import BenchmarkConfig
from quantfin.techniques.integration                      import IntegrationTechnique
from quantfin.techniques.fft                              import FFTTechnique
from quantfin.techniques.monte_carlo                      import MonteCarloTechnique

hyperbolic_params = {"alpha": 15.0, "beta": -5.0, "delta": 0.5, "mu": 0.0}
model = HyperbolicModel(params=hyperbolic_params)

stock = Stock(spot=100.0, dividend=0.01)
rate = Rate(rate=0.03)
options_to_price = [
    Option(strike=100.0, maturity=1.0, option_type=OptionType.CALL),
    Option(strike=100.0, maturity=1.0, option_type=OptionType.PUT),
]

config = BenchmarkConfig(
    name="Hyperbolic Model",
    model=model,
    model_params=hyperbolic_params,
    techniques=[
        (IntegrationTechnique(), "Integration"),
        (FFTTechnique(n=14), "FFT"),
        (MonteCarloTechnique(n_paths=20000, n_steps=100, seed=42), "MC")
    ],
    stock=stock,
    rate=rate,
    options=options_to_price,
)