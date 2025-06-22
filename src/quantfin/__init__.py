"""
quantfin - A Python Quantitative Finance Library
=================================================
This library provides a comprehensive suite of tools for pricing financial
derivatives, calibrating models, and analyzing market data.
"""

# Core Data Structures (Atoms) & Results
from .atoms import ExerciseStyle, Option, OptionType, Rate, Stock, ZeroCouponBond

# Calibration & Workflows
from .calibration import Calibrator, VolatilitySurface

# Models
from .models import (
    BatesModel,
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
    SABRJumpModel,
    SABRModel,
    VarianceGammaModel,
    VasicekModel,
)
from .parity import ImpliedRateModel, ParityModel

# Techniques
from .techniques import (
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
    "SABRJumpModel",
    "SABRModel",
    "VarianceGammaModel",
    "VasicekModel",
    # Techniques
    "ClosedFormTechnique",
    "CRRTechnique",
    "FFTTechnique",
    "IntegrationTechnique",
    "LeisenReimerTechnique",
    "MonteCarloTechnique",
    "PDETechnique",
    "TOPMTechnique",
    # Workflows
    "BacktestWorkflow",
    "Calibrator",
    "DailyWorkflow",
    "VolatilitySurface",
]
