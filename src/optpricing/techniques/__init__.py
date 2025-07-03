from __future__ import annotations

__doc__ = """
The `techniques` package provides the various numerical and analytical
methods for pricing options.
"""

from .american_monte_carlo import AmericanMonteCarloTechnique
from .base import BaseTechnique, GreekMixin, IVMixin, PricingResult
from .closed_form import ClosedFormTechnique
from .crr import CRRTechnique
from .fft import FFTTechnique
from .integration import IntegrationTechnique
from .leisen_reimer import LeisenReimerTechnique
from .monte_carlo import MonteCarloTechnique
from .pde import PDETechnique
from .topm import TOPMTechnique

__all__ = [
    "BaseTechnique",
    "PricingResult",
    "GreekMixin",
    "IVMixin",
    "ClosedFormTechnique",
    "FFTTechnique",
    "IntegrationTechnique",
    "CRRTechnique",
    "LeisenReimerTechnique",
    "TOPMTechnique",
    "PDETechnique",
    "MonteCarloTechnique",
    "AmericanMonteCarloTechnique",
]
