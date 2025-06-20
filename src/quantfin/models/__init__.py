from __future__ import annotations

__doc__ = """
The `models` package contains all financial models for option pricing.

It provides the abstract `BaseModel` and a suite of concrete implementations,
from the standard Black-Scholes-Merton to advanced stochastic volatility
and jump-diffusion models.
"""

# Base classes and utilities
from .base import BaseModel, CF, PDECoeffs, ParamValidator

# Concrete model implementations
from .bates import BatesModel
from .blacks_approx import BlacksApproxModel
from .bsm import BSMModel
from .cev import CEVModel
from .cgmy import CGMYModel
from .dupire_local import DupireLocalVolModel
from .heston import HestonModel
from .hyperbolic import HyperbolicModel
from .kou import KouModel
from .merton_jump import MertonJumpModel
from .nig import NIGModel
from .perpetual_put import PerpetualPutModel
from .sabr_jump import SABRJumpModel # This is correct
from .sabr import SABRModel
from .vg import VarianceGammaModel
from .vasicek import VasicekModel
from .cir import CIRModel
__all__ = [
    # Base
    "BaseModel", "CF", "PDECoeffs", "ParamValidator",
    # Models
    "BatesModel",
    "BlacksApproxModel",
    "BSMModel",
    "CEVModel",
    "CGMYModel",
    "DupireLocalVolModel",
    "HestonModel",
    "HyperbolicModel",
    "KouModel",
    "MertonJumpModel",
    "NIGModel",
    "PerpetualPutModel",
    "SABRJumpModel",
    "SABRModel",
    "VarianceGammaModel",
    "VasicekModel",
    "CIRModel",
]