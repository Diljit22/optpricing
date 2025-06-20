
__doc__ = """
The `techniques` package provides the various numerical and analytical
methods for pricing options.
"""
# Import from sub-modules to make them available at the package level.
from .base import BaseTechnique, PricingResult, GreekMixin, IVMixin
from .closed_form import ClosedFormTechnique
from .fft import FFTTechnique
from .integration import IntegrationTechnique
from .lattice import CRRTechnique, LeisenReimerTechnique, TOPMTechnique
from .pde import PDETechnique

# Finally, import the MonteCarloTechnique which depends on the kernels
from .monte_carlo import MonteCarloTechnique


# Define the public API for this package.
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
]