"""
quantfin - A Python Quantitative Finance Library
=================================================
This library provides a comprehensive suite of tools for pricing financial
derivatives, calibrating models, and analyzing market data.
"""

# --- Core Data Structures (Atoms) ---
from .atoms import Option, OptionType, ExerciseStyle, Rate, Stock, ZeroCouponBond

# --- Core Pricing Result ---
from .techniques.base import PricingResult

# --- Models ---
from .models import (
    BSMModel, MertonJumpModel, HestonModel, BatesModel, SABRModel, CEVModel,
    KouModel, VarianceGammaModel, NIGModel, CGMYModel, HyperbolicModel,
    DupireLocalVolModel, VasicekModel, CIRModel,
)

from .parity import(
    ParityModel, ImpliedRateModel,
)

# --- Techniques ---
from .techniques import (
    ClosedFormTechnique, IntegrationTechnique, FFTTechnique, MonteCarloTechnique,
    PDETechnique, CRRTechnique, LeisenReimerTechnique, TOPMTechnique
)

# --- Calibration & Workflows ---
from .calibration import Calibrator, VolatilitySurface
from .workflows import DailyWorkflow, BacktestWorkflow

# --- Define the public API for the top-level package ---
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