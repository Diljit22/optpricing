import math

import pytest

from optpricing.parity import ParityModel

# Test data based on a known BSM model output
# S=100, K=105, T=1, r=0.05, q=0.01, vol=0.2
# Call Price = 7.49170, Put Price = 8.36581
TEST_PARAMS = {
    "spot": 100.0,
    "strike": 105.0,
    "t": 1.0,
    "r": 0.05,
    "q": 0.01,
}
CALL_PRICE = 7.49170
PUT_PRICE = 8.36581


@pytest.fixture
def model():
    """Provides a ParityModel instance for tests."""
    return ParityModel(params={})


def test_parity_get_put_from_call(model):
    """
    Tests calculating a put price given a call price.
    """
    calculated_put = model._closed_form_impl(
        **TEST_PARAMS, call=True, option_price=CALL_PRICE
    )
    assert calculated_put == pytest.approx(PUT_PRICE, abs=1e-3)


def test_parity_get_call_from_put(model):
    """
    Tests calculating a call price given a put price.
    """
    calculated_call = model._closed_form_impl(
        **TEST_PARAMS, call=False, option_price=PUT_PRICE
    )
    assert calculated_call == pytest.approx(CALL_PRICE, abs=1e-3)


def test_price_bounds_call(model):
    """
    Tests the no-arbitrage price bounds for a call option.
    """
    discounted_strike = TEST_PARAMS["strike"] * math.exp(
        -TEST_PARAMS["r"] * TEST_PARAMS["t"]
    )
    expected_lower = max(0, TEST_PARAMS["spot"] - discounted_strike)
    expected_upper = TEST_PARAMS["spot"]

    lower, upper = model.price_bounds(
        spot=TEST_PARAMS["spot"],
        strike=TEST_PARAMS["strike"],
        r=TEST_PARAMS["r"],
        t=TEST_PARAMS["t"],
        call=True,
        option_price=CALL_PRICE,
    )

    assert lower == pytest.approx(expected_lower)
    assert upper == pytest.approx(expected_upper)


def test_price_bounds_put(model):
    """
    Tests the no-arbitrage price bounds for a put option.
    """
    discounted_strike = TEST_PARAMS["strike"] * math.exp(
        -TEST_PARAMS["r"] * TEST_PARAMS["t"]
    )
    expected_lower = max(0, discounted_strike - TEST_PARAMS["spot"])
    expected_upper = discounted_strike

    lower, upper = model.price_bounds(
        spot=TEST_PARAMS["spot"],
        strike=TEST_PARAMS["strike"],
        r=TEST_PARAMS["r"],
        t=TEST_PARAMS["t"],
        call=False,
        option_price=PUT_PRICE,
    )

    assert lower == pytest.approx(expected_lower)
    assert upper == pytest.approx(expected_upper)


@pytest.mark.parametrize("call", [True, False])
@pytest.mark.parametrize("strike", [95.0, 105.0])
def test_price_bounds_with_dividend_yield(model, call, strike):
    """
    Merton (1973) bounds with a dividend yield: the lower bound is the
    sigma -> 0 limit of BSM, max(S e^{-qT} - K e^{-rT}, 0) for a call, and the
    call is capped by S e^{-qT}. Regression: q was ignored (call lower bound
    7.8077 instead of 6.6890 at S=100, K=95, T=0.75, r=4 %, q=1.5 %).
    """
    from optpricing import BSMModel

    S, r, q, T = 100.0, 0.04, 0.015, 0.75
    lower, upper = model.price_bounds(
        spot=S, strike=strike, r=r, q=q, t=T, call=call, option_price=0.0
    )
    zero_vol = BSMModel(params={"sigma": 1e-6}).price_closed_form(
        spot=S, strike=strike, r=r, q=q, t=T, call=call
    )
    assert lower == pytest.approx(zero_vol, abs=1e-9)
    assert upper == pytest.approx(
        S * math.exp(-q * T) if call else strike * math.exp(-r * T)
    )
    for sigma in (0.05, 0.3, 2.0):
        price = BSMModel(params={"sigma": sigma}).price_closed_form(
            spot=S, strike=strike, r=r, q=q, t=T, call=call
        )
        assert lower - 1e-12 <= price <= upper + 1e-12


def test_lower_bound_rate(model):
    """
    Tests the calculation of the minimum risk-free rate to avoid arbitrage.
    """
    rate = model.lower_bound_rate(
        call_price=CALL_PRICE,
        put_price=PUT_PRICE,
        spot=TEST_PARAMS["spot"],
        strike=TEST_PARAMS["strike"],
        t=TEST_PARAMS["t"],
    )
    expected_val = TEST_PARAMS["strike"] / (
        TEST_PARAMS["spot"] - CALL_PRICE + PUT_PRICE
    )
    expected_rate = math.log(expected_val) / TEST_PARAMS["t"]
    assert rate == pytest.approx(expected_rate)


def test_lower_bound_rate_arbitrage(model):
    """
    Tests that lower_bound_rate raises ValueError when an arbitrage exists.
    """
    with pytest.raises(ValueError, match="Arbitrage exists"):
        model.lower_bound_rate(
            call_price=110.0,
            put_price=5.0,
            spot=TEST_PARAMS["spot"],
            strike=TEST_PARAMS["strike"],
            t=TEST_PARAMS["t"],
        )
