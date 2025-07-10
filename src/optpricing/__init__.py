"""
optpricing - A Python Quantitative Finance Library
=================================================

This library provides a comprehensive suite of tools for pricing financial
derivatives, calibrating models, and analyzing market data.
"""

from importlib.metadata import version as _v

__version__ = _v(__name__)

# Core Data Structures (Atoms)
from .atoms import (
    ExerciseStyle,
    Option,
    OptionType,
    Rate,
    Stock,
    ZeroCouponBond,
)

# Calibration
from .calibration import Calibrator, VolatilitySurface

# Models
from .models import (
    BatesModel,
    BlacksApproxModel,
    BSMModel,
    CEVModel,
    CGMYModel,
    CIRModel,
    DupireLocalVolModel,
    HestonModel,
    HyperbolicModel,
    KouModel,
    MertonJumpModel,
    NIGModel,
    PerpetualPutModel,
    SABRJumpModel,
    SABRModel,
    VarianceGammaModel,
    VasicekModel,
)
from .parity import ImpliedRateModel, ParityModel

# Techniques
from .techniques import (
    AmericanMonteCarloTechnique,
    ClosedFormTechnique,
    CRRTechnique,
    FFTTechnique,
    IntegrationTechnique,
    LeisenReimerTechnique,
    MonteCarloTechnique,
    PDETechnique,
    TOPMTechnique,
)
from .techniques.base import PricingResult
from .workflows import BacktestWorkflow, DailyWorkflow

# Define the public API for the top level package
__all__ = [
    # Atoms & Results
    "ExerciseStyle",
    "Option",
    "OptionType",
    "PricingResult",
    "Rate",
    "Stock",
    "ZeroCouponBond",
    # Models
    "BatesModel",
    "BlacksApproxModel",
    "BSMModel",
    "CEVModel",
    "CGMYModel",
    "CIRModel",
    "DupireLocalVolModel",
    "HestonModel",
    "HyperbolicModel",
    "ImpliedRateModel",
    "KouModel",
    "MertonJumpModel",
    "NIGModel",
    "ParityModel",
    "PerpetualPutModel",
    "SABRJumpModel",
    "SABRModel",
    "VarianceGammaModel",
    "VasicekModel",
    # Techniques
    "AmericanMonteCarloTechnique",
    "ClosedFormTechnique",
    "CRRTechnique",
    "FFTTechnique",
    "IntegrationTechnique",
    "LeisenReimerTechnique",
    "MonteCarloTechnique",
    "PDETechnique",
    "TOPMTechnique",
    # Workflows & Calibration
    "BacktestWorkflow",
    "Calibrator",
    "DailyWorkflow",
    "VolatilitySurface",
]
