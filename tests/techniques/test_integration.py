from unittest.mock import MagicMock, patch

import pytest

from optpricing.atoms import Option, OptionType, Rate, Stock
from optpricing.models import BSMModel
from optpricing.techniques import IntegrationTechnique


# Common setup for tests
@pytest.fixture
def setup():
    option = Option(strike=105, maturity=1.0, option_type=OptionType.CALL)
    stock = Stock(spot=100)
    model = BSMModel(params={"sigma": 0.2})
    rate = Rate(rate=0.05)
    return option, stock, model, rate


def test_integration_model_type_check(setup):
    """
    Tests that the IntegrationTechnique raises a TypeError if the model
    does not support a characteristic function.
    """
    option, stock, _, rate = setup

    non_cf_model = MagicMock()
    non_cf_model.supports_cf = False
    non_cf_model.name = "NoCFModel"

    technique = IntegrationTechnique()
    with pytest.raises(TypeError, match="does not support a characteristic function"):
        technique.price(option, stock, non_cf_model, rate)


def test_integration_price_convergence(setup):
    """
    Tests that the integration price converges to the BSM analytical price.
    """
    option, stock, model, rate = setup

    bsm_price = model.price_closed_form(
        spot=stock.spot,
        strike=option.strike,
        r=rate.get_rate(option.maturity),
        q=stock.dividend,
        t=option.maturity,
        call=(option.option_type is OptionType.CALL),
    )

    technique = IntegrationTechnique()
    integration_price = technique.price(option, stock, model, rate).price

    assert integration_price == pytest.approx(bsm_price, abs=1e-6)


def test_free_delta_accuracy(setup):
    """
    Tests that the 'free' delta from integration is close to the analytical BSM delta.
    """
    option, stock, model, rate = setup

    bsm_delta = model.delta_analytic(
        spot=stock.spot,
        strike=option.strike,
        r=rate.get_rate(option.maturity),
        q=stock.dividend,
        t=option.maturity,
        call=(option.option_type is OptionType.CALL),
    )

    technique = IntegrationTechnique()

    technique.price(option, stock, model, rate)
    integration_delta = technique.delta(option, stock, model, rate)

    assert integration_delta == pytest.approx(bsm_delta, abs=1e-6)


@patch("optpricing.techniques.base.greek_mixin.GreekMixin.delta")
def test_delta_fallback_mechanism(mock_super_delta, setup):
    """
    Tests that the delta method falls back to the superclass's implementation
    if the cached delta is NaN.
    """
    option, stock, model, rate = setup
    technique = IntegrationTechnique()

    # Mock the internal pricing method to return a NaN delta
    nan_results = {"price": 10.0, "delta": float("nan")}
    with patch.object(technique, "_price_and_delta", return_value=nan_results):
        technique.delta(option, stock, model, rate)

        # Assert that the superclass's (finite difference) delta was called
        mock_super_delta.assert_called_once()
