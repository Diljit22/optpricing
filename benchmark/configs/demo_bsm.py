import numpy as np
from atoms.option import Option, OptionType
from atoms.stock import Stock
from atoms.rate import Rate
from models.bsm import BSMModel
from techniques.closed_form import ClosedFormTechnique
from techniques.crr import CRRLatticeTechnique
from techniques.leisen_reimer import LeisenReimerTechnique
from techniques.pde import PDESolverTechnique
from techniques.topm import TOPMLatticeTechnique
from .base_config import BenchmarkConfig
from techniques.integration import IntegrationTechnique
from techniques.fft import FFTTechnique
from techniques.monte_carlo import MonteCarloTechnique


S0, vol, r, q = 100.0, 0.2, 0.03, 0.01
bsm_model = BSMModel(params={"sigma": vol})
stock = Stock(spot=S0, volatility=vol, dividend=q)
rate = Rate(rate=r)

options_to_price = [
    Option(strike=100.0, maturity=1.0, option_type=OptionType.CALL),
    Option(strike=100.0, maturity=1.0, option_type=OptionType.PUT),
]

techniques_to_run = [
    (ClosedFormTechnique(use_analytic_greeks=True), "Analytic"),
    (ClosedFormTechnique(use_analytic_greeks=False), "Finite Diff"),
    (IntegrationTechnique(), "Integration"),
    (FFTTechnique(n=14), "FFT"),
    (PDESolverTechnique(), "PDE"),
    (CRRLatticeTechnique(steps=300), "CRR"),
    (LeisenReimerTechnique(steps=300), "LR"),
    (TOPMLatticeTechnique(steps=300), "TOPM"),
    (MonteCarloTechnique(n_paths=50000, n_steps=500, seed=42), "MC"),
    #(LewisFFTTechnique(n=14), "FFT (Lewis)")
]

config = BenchmarkConfig(
    name="Black-Scholes-Merton Model vs. All Techniques",
    model=bsm_model,
    model_params={"sigma": vol}, # Redundant but good for display
    techniques=techniques_to_run,
    stock=stock,
    rate=rate,
    options=options_to_price,
)