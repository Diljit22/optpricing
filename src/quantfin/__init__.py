"""
quantfin - A Python Quantitative Finance Library
=================================================
This library provides a comprehensive suite of tools for pricing financial
derivatives, calibrating models, and analyzing market data.
"""

# Core Data Structures (Atoms)
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
    SABRModel,
    VarianceGammaModel,
    VasicekModel,
)
from .parity import (
    ImpliedRateModel,
    ParityModel,
)

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

# Core Pricing Result
from .techniques.base import PricingResult
from .workflows import BacktestWorkflow, DailyWorkflow

# Define the public API for the top-level package
__all__ = [
    # Atoms & Results
    "Option", "OptionType", "ExerciseStyle", "Rate", "Stock", "ZeroCouponBond", "PricingResult",
    # Models
    "BSMModel", "MertonJumpModel", "HestonModel", "BatesModel", "SABRModel", "CEVModel",
    "KouModel", "VarianceGammaModel", "NIGModel", "CGMYModel", "HyperbolicModel",
    "DupireLocalVolModel", "VasicekModel", "CIRModel", "ParityModel", "ImpliedRateModel",
    # Techniques
    "ClosedFormTechnique", "IntegrationTechnique", "FFTTechnique", "MonteCarloTechnique",
    "PDETechnique", "CRRTechnique", "LeisenReimerTechnique", "TOPMTechnique",
    # Workflows
    "Calibrator", "VolatilitySurface", "DailyWorkflow", "BacktestWorkflow",
]
