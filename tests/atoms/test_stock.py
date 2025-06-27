import pytest

from optpricing.atoms import Stock


def test_stock_creation():
    """
    Tests successful creation of a Stock instance with all parameters.
    """
    stock = Stock(
        spot=150,
        volatility=0.2,
        dividend=0.01,
        discrete_dividends=[2.0],
        ex_div_times=[0.5],
    )
    assert stock.spot == 150
    assert stock.volatility == 0.2
    assert stock.dividend == 0.01
    assert stock.discrete_dividends == [2.0]
    assert stock.ex_div_times == [0.5]


def test_stock_minimal_creation():
    """
    Tests successful creation of a Stock instance with only the spot price.
    """
    stock = Stock(spot=150)
    assert stock.spot == 150
    assert stock.volatility is None
    assert stock.dividend == 0.0
    assert stock.discrete_dividends is None
    assert stock.ex_div_times is None


@pytest.mark.parametrize("invalid_spot", [0, -150])
def test_stock_invalid_spot(invalid_spot):
    """
    Tests that creating a stock with a non-positive spot price raises ValueError.
    """
    with pytest.raises(ValueError, match="Spot price must be positive"):
        Stock(spot=invalid_spot)
