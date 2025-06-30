from unittest.mock import patch

import pytest

from optpricing.atoms import Option, OptionType, Rate, Stock
from optpricing.models import BSMModel
from optpricing.techniques import LeisenReimerTechnique


# Common setup for tests
@pytest.fixture
def setup():
    option = Option(strike=100, maturity=1.0, option_type=OptionType.CALL)
    stock = Stock(spot=100, volatility=0.2)
    model = BSMModel(params={"sigma": 0.2})
    rate = Rate(rate=0.05)
    return option, stock, model, rate


@patch("optpricing.techniques.leisen_reimer._lr_pricer")
def test_lr_calls_kernel(mock_pricer, setup):
    """
    Tests that the LeisenReimerTechnique class correctly calls the _lr_pricer kernel.
    """
    option, stock, model, rate = setup

    mock_pricer.return_value = {"price": 10.45058}

    technique = LeisenReimerTechnique(steps=101, is_american=False)
    result = technique.price(option, stock, model, rate)

    mock_pricer.assert_called_once()

    args, kwargs = mock_pricer.call_args
    assert kwargs["S0"] == stock.spot
    assert kwargs["K"] == option.strike
    assert kwargs["T"] == option.maturity
    assert kwargs["sigma"] == model.params["sigma"]
    assert kwargs["N"] == 101
    assert kwargs["is_am"] is False
    assert kwargs["is_call"] is True

    assert result.price == 10.45058
