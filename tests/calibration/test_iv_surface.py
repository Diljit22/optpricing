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
    # Should succeed
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


@patch("optpricing.calibration.iv_surface.VolatilitySurface._calculate_ivs")
def test_calculate_model_iv(mock_calculate, setup):
    """
    Tests that calculate_model_iv calls the technique and then the IV solver.
    """
    surface, stock, rate = setup

    # Mock the model and technique
    model = MagicMock()
    technique = MagicMock()
    # Make the mock technique return different prices for each option
    technique.price.side_effect = [MagicMock(price=10.1), MagicMock(price=2.1)]

    mock_calculate.return_value = np.array([0.21, 0.21])  # Dummy IVs

    surface.calculate_model_iv(stock, rate, model, technique)

    # Assert that the pricing technique was called for each row of data
    assert technique.price.call_count == len(surface.data)

    # Assert that the IV solver was called with the model prices
    mock_calculate.assert_called_once()
    model_prices_series = pd.Series([10.1, 2.1], index=surface.data.index)
    pd.testing.assert_series_equal(
        mock_calculate.call_args[0][2], model_prices_series, check_names=False
    )
    assert "iv" in surface.surface.columns
