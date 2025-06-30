from unittest.mock import MagicMock

import numpy as np
import pytest

from optpricing.atoms import Option, OptionType, Rate, Stock
from optpricing.models import BSMModel
from optpricing.techniques import FFTTechnique


# Common setup for tests
@pytest.fixture
def setup():
    option = Option(strike=105, maturity=1.0, option_type=OptionType.CALL)
    stock = Stock(spot=100)
    model = BSMModel(params={"sigma": 0.2})
    rate = Rate(rate=0.05)
    return option, stock, model, rate


def test_fft_model_type_check(setup):
    """
    Tests that the FFTTechnique raises a TypeError if the model
    does not support a characteristic function.
    """
    option, stock, _, rate = setup

    # Create a mock model that does not support CF
    non_cf_model = MagicMock()
    non_cf_model.supports_cf = False
    non_cf_model.name = "NoCFModel"

    technique = FFTTechnique()
    with pytest.raises(TypeError, match="does not support a characteristic function"):
        technique.price(option, stock, non_cf_model, rate)


def test_fft_price_convergence(setup):
    """
    Tests that the FFT price converges to the BSM analytical price.
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

    # Use a reasonably fine grid for the test
    technique = FFTTechnique(n=14)
    fft_price = technique.price(option, stock, model, rate).price

    assert fft_price == pytest.approx(bsm_price, abs=1e-3)


def test_fft_put_call_parity(setup):
    """
    Tests that the FFT prices for a call and a put satisfy put-call parity.
    """
    option, stock, model, rate = setup
    put_option = option.parity_counterpart()

    technique = FFTTechnique(n=14)

    call_price = technique.price(option, stock, model, rate).price
    put_price = technique.price(put_option, stock, model, rate).price

    S, K, T = stock.spot, option.strike, option.maturity
    r, q = rate.get_rate(T), stock.dividend

    parity_lhs = call_price - put_price
    parity_rhs = S * np.exp(-q * T) - K * np.exp(-r * T)

    assert parity_lhs == pytest.approx(parity_rhs, abs=1e-5)
