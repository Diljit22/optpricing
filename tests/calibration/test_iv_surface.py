from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from optpricing.atoms import Rate, Stock
from optpricing.calibration import VolatilitySurface


@pytest.fixture
def sample_option_data():
    """Provides a sample DataFrame of option data."""
    return pd.DataFrame(
        {
            "strike": [95, 105],
            "maturity": [1.0, 1.0],
            "marketPrice": [10.0, 2.0],
            "optionType": ["call", "call"],
            "expiry": [pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-01")],
        }
    )


@pytest.fixture
def setup(sample_option_data):
    """Provides a standard setup for tests."""
    stock = Stock(spot=100)
    rate = Rate(rate=0.05)
    surface = VolatilitySurface(sample_option_data)
    return surface, stock, rate


def test_volatility_surface_initialization(sample_option_data):
    """
    Tests that the class initializes correctly and validates columns.
    """
    VolatilitySurface(sample_option_data)

    # Should fail if a column is missing
    with pytest.raises(ValueError, match="missing one of the required columns"):
        VolatilitySurface(sample_option_data.drop(columns=["strike"]))


@patch("optpricing.calibration.iv_surface.VolatilitySurface._calculate_ivs")
def test_calculate_market_iv(mock_calculate, setup):
    """
    Tests that calculate_market_iv calls the IV solver with market prices.
    """
    surface, stock, rate = setup

    mock_calculate.return_value = np.array([0.2, 0.2])  # Dummy IVs

    surface.calculate_market_iv(stock, rate)

    mock_calculate.assert_called_once()
    # Check that the second argument passed was the 'marketPrice' series
    pd.testing.assert_series_equal(
        mock_calculate.call_args[0][2], surface.data["marketPrice"], check_names=False
    )
    assert "iv" in surface.surface.columns


@patch("optpricing.calibration.iv_surface.price_options_vectorized")
def test_calculate_model_iv(mock_vectorized_pricer, setup):
    """
    Tests that calculate_model_iv calls the vectorized pricer and then the IV solver.
    """
    surface, stock, rate = setup
    model = MagicMock()

    mock_vectorized_pricer.return_value = np.array([10.1, 2.1])
    surface.calculate_model_iv(stock, rate, model)
    mock_vectorized_pricer.assert_called_once()
