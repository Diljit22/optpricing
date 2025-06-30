from unittest.mock import patch

import pytest

from optpricing.atoms import Option, OptionType, Rate, Stock
from optpricing.models import BSMModel
from optpricing.techniques import CRRTechnique


# Common setup for tests
@pytest.fixture
def setup():
    option = Option(strike=100, maturity=1.0, option_type=OptionType.CALL)
    stock = Stock(spot=100, volatility=0.2)
    model = BSMModel(params={"sigma": 0.2})
    rate = Rate(rate=0.05)
    return option, stock, model, rate


@patch("optpricing.techniques.crr._crr_pricer")
def test_crr_calls_kernel(mock_pricer, setup):
    """
    Tests that the CRRTechnique class correctly calls the _crr_pricer kernel.
    """
    option, stock, model, rate = setup

    # The mock will return a dummy dictionary that the technique expects
    mock_pricer.return_value = {"price": 10.45058}

    technique = CRRTechnique(steps=101, is_american=False)
    result = technique.price(option, stock, model, rate)

    # Assert that the kernel was called once
    mock_pricer.assert_called_once()

    # Assert that the arguments passed to the kernel are correct
    args, kwargs = mock_pricer.call_args
    assert kwargs["S0"] == stock.spot
    assert kwargs["K"] == option.strike
    assert kwargs["T"] == option.maturity
    assert kwargs["sigma"] == model.params["sigma"]
    assert kwargs["N"] == 101
    assert kwargs["is_am"] is False
    assert kwargs["is_call"] is True

    # Assert that the price from the kernel is returned correctly
    assert result.price == 10.45058
