from unittest.mock import MagicMock

import pytest

from optpricing.calibration.technique_selector import select_fastest_technique
from optpricing.techniques import (
    ClosedFormTechnique,
    COSTechnique,
    MonteCarloTechnique,
)


def test_select_closed_form():
    """
    Tests that ClosedFormTechnique is selected for models that support it.
    """
    model = MagicMock()
    model.has_closed_form = True
    technique = select_fastest_technique(model)
    assert isinstance(technique, ClosedFormTechnique)


def test_select_cos():
    """
    COS is selected for models that support a CF but no closed form: it is as
    accurate as FFT and 4-100x faster per option (0.1-0.5 ms vs 0.4-55 ms).
    """
    model = MagicMock()
    model.has_closed_form = False
    model.supports_cf = True
    technique = select_fastest_technique(model)
    assert isinstance(technique, COSTechnique)


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
