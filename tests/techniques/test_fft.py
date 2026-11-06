from unittest.mock import MagicMock

import numpy as np
import pytest

from optpricing.atoms import Option, OptionType, Rate, Stock
from optpricing.models import BSMModel, VarianceGammaModel
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


@pytest.mark.parametrize("sigma", [0.1, 0.2, 0.4])
@pytest.mark.parametrize("T", [1 / 365, 7 / 365, 0.1, 1.0, 5.0])
def test_fft_matches_black_scholes_at_short_maturities(sigma, T):
    """
    Regression (P5-F17): the frequency grid stopped at u = N eta = 512 whatever
    the maturity, and the price was interpolated on a log-strike grid with
    spacing 0.012; at one day and sigma = 10 % the error was 4.4e-2 (6e-4 at
    T = 0.1, 2e-5 at T = 1). Closed-form Black-Scholes is the reference.
    """
    stock, rate = Stock(spot=100.0, dividend=0.01), Rate(rate=0.03)
    model = BSMModel(params={"sigma": sigma})
    for K in (90.0, 99.0, 100.0, 100.3, 101.0, 110.0):
        for kind in (OptionType.CALL, OptionType.PUT):
            option = Option(strike=K, maturity=T, option_type=kind)
            exact = model.price_closed_form(
                spot=100.0, strike=K, r=0.03, q=0.01, t=T, call=kind is OptionType.CALL
            )
            price = FFTTechnique().price(option, stock, model, rate).price
            assert price == pytest.approx(exact, abs=1e-8), (K, kind)


def test_fft_warns_when_the_frequency_grid_is_capped():
    """Variance Gamma over one day: |phi| ~ u^(-2T/nu) has barely decayed at
    N = 2^18, so FFT warns; the price is still within 2e-5 of the normal-mixture
    truth (put integrated over the gamma quantiles, call by parity; Madan, Carr &
    Chang 1998)."""
    option = Option(strike=100.0, maturity=1 / 365, option_type=OptionType.CALL)
    stock, rate = Stock(spot=100.0, dividend=0.01), Rate(rate=0.03)
    model = VarianceGammaModel(params={"sigma": 0.2, "nu": 0.1, "theta": -0.14})
    with pytest.warns(RuntimeWarning, match="FFT frequency grid truncated"):
        price = FFTTechnique().price(option, stock, model, rate).price
    assert price == pytest.approx(0.1358377801533, abs=2e-5)
