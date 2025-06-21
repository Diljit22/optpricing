from quantfin.models.base import BaseModel
from quantfin.techniques.closed_form import ClosedFormTechnique
from quantfin.techniques.fft import FFTTechnique
from quantfin.techniques.monte_carlo import MonteCarloTechnique


def select_fastest_technique(model: BaseModel):
    """
    Selects the fastest available pricing technique for a given model.
    The order of preference is: Closed-Form > FFT > Integration > Monte Carlo.
    """
    if model.has_closed_form:
        return ClosedFormTechnique()
    if model.supports_cf:
        return FFTTechnique(n=12)
    if model.supports_sde:
        return MonteCarloTechnique(n_paths=5000, n_steps=50, antithetic=True)
    raise TypeError(f"No suitable pricing technique found for model '{model.name}'")
