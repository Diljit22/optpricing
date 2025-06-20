# src/quantfin/__init__.py (Cleaner Version)

"""
qf - A Python Quantitative Finance Library
===========================================
This top-level package provides convenient access to the most common
components of the library.
"""

# --- Core Data Structures (Atoms) ---
from .atoms import Option, OptionType, ExerciseStyle, Rate, Stock, ZeroCouponBond

# --- Core Pricing Result ---
from .techniques.base import PricingResult

# --- Models ---
# Import the most common models directly for ease of use
from .models import (
    BSMModel, MertonJumpModel, HestonModel, BatesModel, SABRModel,
    VasicekModel, CIRModel
)

# --- Techniques ---
# Import the most common techniques directly for ease of use
from .techniques import (
    ClosedFormTechnique, IntegrationTechnique, FFTTechnique,
    MonteCarloTechnique, PDETechnique, CRRTechnique
)

# --- Define the public API for the top-level package ---
__all__ = [
    # Atoms
    "Option", "OptionType", "ExerciseStyle", "Rate", "Stock", "ZeroCouponBond",
    "PricingResult",

    # Common Models
    "BSMModel", "MertonJumpModel", "HestonModel", "BatesModel", "SABRModel",
    "VasicekModel", "CIRModel",

    # Common Techniques
    "ClosedFormTechnique", "IntegrationTechnique", "FFTTechnique",
    "MonteCarloTechnique", "PDETechnique", "CRRTechnique",
]