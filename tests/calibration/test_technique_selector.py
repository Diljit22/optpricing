from unittest.mock import MagicMock

import pytest

from optpricing.calibration.technique_selector import select_fastest_technique
from optpricing.techniques import ClosedFormTechnique, FFTTechnique, MonteCarloTechnique


def test_select_closed_form():
    """
    Tests that ClosedFormTechnique is selected for models that support it.
    """
    model = MagicMock()
    model.has_closed_form = True
    technique = select_fastest_technique(model)
    assert isinstance(technique, ClosedFormTechnique)


def test_select_fft():
    """
    Tests that FFTTechnique is selected for models that support CF but not closed-form.
    """
    model = MagicMock()
    model.has_closed_form = False
    model.supports_cf = True
    technique = select_fastest_technique(model)
    assert isinstance(technique, FFTTechnique)


def test_select_monte_carlo():
    """
    Tests that MonteCarloTechnique is selected for models that only support SDE.
    """
    model = MagicMock()
    model.has_closed_form = False
    model.supports_cf = False
    model.supports_sde = True
    technique = select_fastest_technique(model)
    assert isinstance(technique, MonteCarloTechnique)


def test_select_no_suitable_technique():
    """
    Tests that a TypeError is raised if no suitable technique is found.
    """
    model = MagicMock()
    model.has_closed_form = False
    model.supports_cf = False
    model.supports_sde = False
    model.name = "UnsupportedModel"

    with pytest.raises(TypeError, match="No suitable pricing technique found"):
        select_fastest_technique(model)
