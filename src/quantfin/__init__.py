"""
qf - A Python Quantitative Finance Library
===========================================

This top-level package provides convenient access to the core components
of the library, allowing users to easily import models, techniques, and
data structures.

Example usage:

    from qf import BSMModel, MonteCarloTechnique, Option, Stock, Rate

    # Define instruments
    stock = Stock(spot=100.0)
    rate = Rate(rate=0.05)
    option = Option(strike=105.0, maturity=1.0, option_type=OptionType.CALL)

    # Instantiate model and technique
    model = BSMModel(params={'sigma': 0.20})
    technique = MonteCarloTechnique(n_paths=50000, seed=42)

    # Price the option
    result = technique.price(option, stock, model, rate)
    print(f"Option Price: {result.price:.4f}")
"""

# --- Core Data Structures (Atoms) ---
from .atoms.option import Option, OptionType, ExerciseStyle
from .atoms.rate import Rate
from .atoms.stock import Stock
from .techniques.base.pricing_result import PricingResult

# --- Pricing Models ---
# Diffusion & Jump-Diffusion
from .models.bsm import BSMModel
from .models.cev import CEVModel
from .models.merton_jump import MertonJumpModel
from .models.kou import KouModel

# Stochastic Volatility
from .models.heston import HestonModel
from .models.sabr import SABRModel
from .models.bates import BatesModel
from .models.sabr_jump import SABRJumpModel

# Pure-Jump Lévy
from .models.vg import VarianceGammaModel
from .models.nig import NIGModel
from .models.cgmy import CGMYModel
from .models.hyperbolic import HyperbolicModel

# Local Volatility
from .models.dupire_local import DupireLocalVolModel

# Specialized & Other
from .models.blacks_approx import BlacksApproxModel
from .models.perpetual_put import PerpetualPutModel

# --- Parity & Implied Models ---
from .parity.parity_model import ParityModel
from .parity.implied_rate import ImpliedRateModel

# --- Pricing Techniques ---
# Fourier Methods
from .techniques.fft import FFTTechnique
from .techniques.integration import IntegrationTechnique
from .techniques.lewis_fft import LewisFFTTechnique

# Simulation Methods
from .techniques.monte_carlo import MonteCarloTechnique

# Tree & PDE Methods
from .techniques.crr import CRRLatticeTechnique
from .techniques.leisen_reimer import LeisenReimerTechnique
from .techniques.topm import TOPMLatticeTechnique
from .techniques.pde import PDESolverTechnique

# Closed-Form & Specialized
from .techniques.closed_form import ClosedFormTechnique
from .techniques.bsm_iv_solver import BSMIVSolver

__all__ = [
    # Atoms
    "Option", "OptionType", "ExerciseStyle",
    "Rate",
    "Stock",
    "PricingResult",

    # Models
    "BSMModel",
    "CEVModel",
    "MertonJumpModel",
    "KouModel",
    "HestonModel",
    "SABRModel",
    "BatesModel",
    "SABRJumpModel",
    "VarianceGammaModel",
    "NIGModel",
    "CGMYModel",
    "HyperbolicModel",
    "DupireLocalVolModel",
    "BlacksApproxModel",
    "PerpetualPutModel",
    "ParityModel",
    "ImpliedRateModel",

    # Techniques
    "FFTTechnique",
    "IntegrationTechnique",
    "LewisFFTTechnique",
    "MonteCarloTechnique",
    "CRRLatticeTechnique",
    "LeisenReimerTechnique",
    "TOPMLatticeTechnique",
    "PDESolverTechnique",
    "ClosedFormTechnique",
    "BSMIVSolver",
]