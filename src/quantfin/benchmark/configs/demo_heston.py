import numpy as np

from quantfin.atoms.option                               import Option, OptionType
from quantfin.atoms.stock                                import Stock
from quantfin.atoms.rate                                 import Rate
from quantfin.models.bsm                                 import BSMModel
from quantfin.models.heston import HestonModel
from quantfin.techniques.closed_form                      import ClosedFormTechnique
from quantfin.techniques.crr                              import CRRLatticeTechnique
from quantfin.techniques.leisen_reimer                    import LeisenReimerTechnique
from quantfin.techniques.pde                              import PDESolverTechnique
from quantfin.techniques.topm                             import TOPMLatticeTechnique
from quantfin.benchmark.configs.base_config               import BenchmarkConfig
from quantfin.techniques.integration                      import IntegrationTechnique
from quantfin.techniques.fft                              import FFTTechnique
from quantfin.techniques.monte_carlo                      import MonteCarloTechnique

v0, kappa, theta, rho, vol_of_vol = 0.04, 2.0, 0.04, -0.7, 0.5
heston_params = {
    "v0": v0, "kappa": kappa, "theta": theta, "rho": rho, "vol_of_vol": vol_of_vol
}
model = HestonModel(params=heston_params)

stock = Stock(spot=100.0, dividend=0.01) # Note: Heston uses v0, not stock.volatility
rate = Rate(rate=0.03)
options_to_price = [
    Option(strike=100.0, maturity=1.0, option_type=OptionType.CALL),
    Option(strike=100.0, maturity=1.0, option_type=OptionType.PUT),
]

config = BenchmarkConfig(
    name="Heston Stochastic Volatility Model",
    model=model,
    model_params=heston_params,
    techniques=[
        (IntegrationTechnique(), "Integration"),
        (FFTTechnique(n=14), "FFT"),
        (MonteCarloTechnique(n_paths=20000, n_steps=100, seed=42), "MC")
        ],
    stock=stock,
    rate=rate,
    options=options_to_price,
    # We must pass v0 to the technique via kwargs because it's not a standard param
    technique_kwargs={"v0": v0}
)