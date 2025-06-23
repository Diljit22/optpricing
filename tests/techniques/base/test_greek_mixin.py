from unittest.mock import MagicMock

import numpy as np
import pytest

from quantfin.atoms import Option, OptionType, Rate, Stock
from quantfin.models import BSMModel
from quantfin.techniques.base import BaseTechnique, GreekMixin, PricingResult


# Create a dummy technique that uses the GreekMixin for testing
class DummyGreekTechnique(BaseTechnique, GreekMixin):
    def __init__(self):
        # The mixin checks for the presence of this attribute for CRN logic
        self.rng = np.random.default_rng(0)
        self.price_func = MagicMock(return_value=PricingResult(price=10.0))

    def price(self, option, stock, model, rate, **kwargs):
        return self.price_func(option, stock, model, rate, **kwargs)


@pytest.fixture
def setup():
    """Provides a standard setup for Greek tests."""
    technique = DummyGreekTechnique()
    option = Option(strike=100, maturity=1.0, option_type=OptionType.CALL)
    stock = Stock(spot=100)
    model = BSMModel(params={"sigma": 0.2})
    rate = Rate(rate=0.05)
    return technique, option, stock, model, rate


def test_delta_calls_price_twice(setup):
    """
    Tests that the delta calculation calls the price method twice.
    """
    technique, option, stock, model, rate = setup
    technique.delta(option, stock, model, rate)
    assert technique.price_func.call_count == 2


def test_gamma_calls_price_thrice(setup):
    """
    Tests that the gamma calculation calls the price method three times.
    """
    technique, option, stock, model, rate = setup
    technique.gamma(option, stock, model, rate)
    assert technique.price_func.call_count == 3


def test_vega_no_sigma(setup):
    """
    Tests that vega returns nan if the model has no 'sigma' parameter.
    """
    technique, option, stock, model, rate = setup

    # Create a model without a 'sigma' param
    no_sigma_model = MagicMock()
    no_sigma_model.params = {"other_param": 0.1}

    vega = technique.vega(option, stock, no_sigma_model, rate)
    assert np.isnan(vega)
    technique.price_func.assert_not_called()
