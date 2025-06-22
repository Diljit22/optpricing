from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from quantfin.atoms import Rate, Stock
from quantfin.dashboard.service import DashboardService


# Fixtures for creating mock data and service instances
@pytest.fixture
def mock_market_data():
    """A sample DataFrame that mimics real market data."""
    return pd.DataFrame(
        {
            "spot_price": [100.0] * 4,
            "optionType": ["call", "call", "put", "put"],
            "strike": [100, 105, 100, 95],
        }
    )


@pytest.fixture
def mock_model_configs():
    """A sample model configuration dictionary."""
    mock_model_class = MagicMock()
    return {"TestModel": {"model_class": mock_model_class}}


@pytest.fixture
def service(mock_model_configs):
    """A DashboardService instance for testing."""
    return DashboardService(
        ticker="TEST",
        snapshot_date="2023-01-01",
        model_configs=mock_model_configs,
    )


# Test Cases
def test_dashboard_service_initialization(service):
    """
    Tests that the service is initialized with the correct attributes.
    """
    assert service.ticker == "TEST"
    assert service.snapshot_date == "2023-01-01"
    assert service._market_data is None
    assert service._stock is None
    assert service._rate is None


@patch("quantfin.dashboard.service.load_market_snapshot")
def test_market_data_property_snapshot(mock_load, service, mock_market_data):
    """
    Tests that the market_data property correctly calls the snapshot loader
    and caches the result.
    """
    mock_load.return_value = mock_market_data
    data = service.market_data
    mock_load.assert_called_once_with("TEST", "2023-01-01")
    pd.testing.assert_frame_equal(data, mock_market_data)
    _ = service.market_data
    mock_load.assert_called_once()


@patch("quantfin.dashboard.service.get_live_option_chain")
def test_market_data_property_live(mock_get_live, mock_model_configs, mock_market_data):
    """
    Tests that the market_data property calls the live fetcher for 'Live Data'.
    """
    service_live = DashboardService("TEST", "Live Data", mock_model_configs)
    mock_get_live.return_value = mock_market_data
    _ = service_live.market_data
    mock_get_live.assert_called_once_with("TEST")


# CORRECTED: Split the single failing test into two focused, passing tests.
@patch("quantfin.dashboard.service.fit_rate_and_dividend", return_value=(0.05, 0.01))
def test_stock_property_caching(mock_fit, service, mock_market_data):
    """
    Tests the .stock property in isolation to verify its logic and caching.
    """
    service._market_data = mock_market_data

    # First access should call the fitting function
    stock = service.stock
    mock_fit.assert_called_once()
    assert isinstance(stock, Stock)
    assert stock.spot == 100.0
    assert stock.dividend == 0.01

    # Second access should use the cached value, not call fit again
    _ = service.stock
    mock_fit.assert_called_once()


@patch("quantfin.dashboard.service.fit_rate_and_dividend", return_value=(0.05, 0.01))
def test_rate_property_caching(mock_fit, service, mock_market_data):
    """
    Tests the .rate property in isolation to verify its logic and caching.
    """
    service._market_data = mock_market_data

    # First access should call the fitting function
    rate = service.rate
    mock_fit.assert_called_once()
    assert isinstance(rate, Rate)
    assert rate.get_rate() == 0.05

    # Second access should use the cached value, not call fit again
    _ = service.rate
    mock_fit.assert_called_once()


@patch("quantfin.dashboard.service.DailyWorkflow")
def test_run_calibrations(mock_workflow, service, mock_market_data):
    """
    Tests that the run_calibrations method correctly instantiates and runs
    the DailyWorkflow for each model.
    """
    mock_instance = MagicMock()
    mock_instance.results = {
        "Status": "Success",
        "Calibrated Params": {"sigma": 0.2},
    }
    mock_workflow.return_value = mock_instance
    service._market_data = mock_market_data

    service.run_calibrations()

    mock_workflow.assert_called_once_with(
        market_data=mock_market_data,
        model_config=service.model_configs["TestModel"],
    )
    mock_instance.run.assert_called_once()
    assert "TestModel" in service.calibrated_models
    assert service.summary_df is not None
    assert service.summary_df.iloc[0]["Status"] == "Success"


@patch("quantfin.dashboard.service.plot_iv_surface_3d")
@patch("quantfin.dashboard.service.plot_smiles_by_expiry")
@patch("quantfin.dashboard.service.select_fastest_technique")
@patch("quantfin.dashboard.service.VolatilitySurface")
def test_get_iv_plots(
    mock_vol_surface,
    mock_select,
    mock_plot_smiles,
    mock_plot_3d,
    service,
    mock_market_data,
):
    """
    Tests that get_iv_plots correctly calls all necessary components to
    generate plots.
    """
    mock_surface_instance = MagicMock()
    mock_surface_instance.calculate_market_iv.return_value = mock_surface_instance
    mock_surface_instance.calculate_model_iv.return_value = mock_surface_instance
    mock_surface_instance.surface = pd.DataFrame()
    mock_vol_surface.return_value = mock_surface_instance
    mock_plot_smiles.return_value = "smile_figure"
    mock_plot_3d.return_value = "surface_figure"

    service._market_data = mock_market_data
    service._stock = Stock(spot=100)
    service._rate = Rate(rate=0.05)
    service.calibrated_models["TestModel"] = MagicMock()

    smile_fig, surface_fig = service.get_iv_plots()

    assert mock_vol_surface.call_count == 2
    mock_select.assert_called_once()
    mock_plot_smiles.assert_called_once()
    mock_plot_3d.assert_called_once()
    assert smile_fig == "smile_figure"
    assert surface_fig == "surface_figure"
