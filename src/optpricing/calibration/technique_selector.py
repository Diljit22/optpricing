from __future__ import annotations

from optpricing.models.base import BaseModel
from optpricing.techniques.closed_form import ClosedFormTechnique
from optpricing.techniques.fft import FFTTechnique
from optpricing.techniques.monte_carlo import MonteCarloTechnique

__doc__ = """
Provides a utility function to select the most efficient pricing technique
for a given financial model based on its supported features.
"""


def select_fastest_technique(model: BaseModel):
    """
    Selects the fastest available pricing technique for a given model.

    The function checks the capabilities of the model in a specific order of
    preference, which generally corresponds to computational speed.

    The order of preference is:
    1. Closed-Form
    2. Fast Fourier Transform (FFT)
    3. Monte Carlo

    Parameters
    ----------
    model : BaseModel
        The financial model for which to select a technique.

    Returns
    -------
    BaseTechnique
        An instance of the fastest suitable pricing technique.

    Raises
    ------
    TypeError
        If no suitable pricing technique can be found for the model.
    """
    if model.has_closed_form:
        return ClosedFormTechnique()
    if model.supports_cf:
        return FFTTechnique(n=12)
    if model.supports_sde:
        return MonteCarloTechnique(n_paths=5000, n_steps=50, antithetic=True)
    raise TypeError(f"No suitable pricing technique found for model '{model.name}'")
