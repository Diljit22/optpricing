import pytest

from optpricing.parity import ImpliedRateModel

# Test data based on a known BSM model output
# S=100, K=100, T=1, r=0.05, q=0.01, vol=0.2
# Call Price = 9.82629, Put Price = 5.94424

TEST_PARAMS = {
    "spot": 100.0,
    "strike": 100.0,
    "t": 1.0,
    "q": 0.01,
    "call_price": 9.82629,
    "put_price": 5.94424,
}
EXPECTED_RATE = 0.05


def test_implied_rate_calculation():
    """
    Tests that the implied rate is calculated correctly for a known set of prices.
    """
    model = ImpliedRateModel(params={})
    implied_rate = model._closed_form_impl(**TEST_PARAMS)
    assert implied_rate == pytest.approx(EXPECTED_RATE, abs=1e-4)


def test_implied_rate_no_dividend():
    """
    Tests implied rate calculation when dividend is zero.
    S=100, K=100, T=1, r=0.05, q=0, vol=0.2
    Call=10.45058, Put=5.57352
    """
    params = {
        "spot": 100.0,
        "strike": 100.0,
        "t": 1.0,
        "q": 0.0,
        "call_price": 10.45058,
        "put_price": 5.57352,
    }
    model = ImpliedRateModel(params={})
    implied_rate = model._closed_form_impl(**params)
    assert implied_rate == pytest.approx(0.05, abs=1e-4)


def test_implied_rate_arbitrage_case():
    """
    Tests that the model raises a ValueError when prices suggest arbitrage
    and a root cannot be bracketed.
    """
    # C - P > S, which is an arbitrage violation.
    arbitrage_params = TEST_PARAMS.copy()
    arbitrage_params["call_price"] = 110.0
    arbitrage_params["put_price"] = 5.0

    model = ImpliedRateModel(params={})
    with pytest.raises(ValueError, match="Unable to bracket a root"):
        model._closed_form_impl(**arbitrage_params)
