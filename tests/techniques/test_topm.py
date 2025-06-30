from unittest.mock import patch

import pytest

from optpricing.atoms import Option, OptionType, Rate, Stock
from optpricing.models import BSMModel
from optpricing.techniques import TOPMTechnique


# Common setup for tests
@pytest.fixture
def setup():
    option = Option(strike=100, maturity=1.0, option_type=OptionType.PUT)
    stock = Stock(spot=100, volatility=0.2)
    model = BSMModel(params={"sigma": 0.2})
    rate = Rate(rate=0.05)
    return option, stock, model, rate


@patch("optpricing.techniques.topm._topm_pricer")
def test_topm_calls_kernel(mock_pricer, setup):
    """
    Tests that the TOPMTechnique class correctly calls the _topm_pricer kernel.
    """
    option, stock, model, rate = setup

    mock_pricer.return_value = {"price": 10.45058}

    technique = TOPMTechnique(steps=101, is_american=False)
    result = technique.price(option, stock, model, rate)

    mock_pricer.assert_called_once()

    args, kwargs = mock_pricer.call_args
    assert kwargs["S0"] == stock.spot
    assert kwargs["K"] == option.strike
    assert kwargs["T"] == option.maturity
    assert kwargs["vol"] == model.params["sigma"]
    assert kwargs["N"] == 101
    assert kwargs["is_am"] is False
    assert kwargs["is_call"] is False

    assert result.price == 10.45058
