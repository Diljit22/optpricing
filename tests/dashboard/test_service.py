from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from optpricing.atoms import Rate, Stock
from optpricing.dashboard.service import DashboardService
from optpricing.models import BSMModel


@pytest.fixture
def mock_model_configs():
    """Sample model configs."""
    mock_bsm_class = BSMModel
    mock_bsm_class.default_params = {"sigma": 0.2}
    return {"BSM": {"model_class": mock_bsm_class, "ticker": "TEST"}}


@pytest.fixture
def service(mock_model_configs):
    """Provides a standard setup for service tests."""
    return DashboardService(
        ticker="TEST", snapshot_date="2023-01-01", model_configs=mock_model_configs
    )


@pytest.fixture
def mock_market_data():
    """A sample DataFrame for market data."""
    return pd.DataFrame(
        {
            "spot_price": [100.0] * 4,
            "optionType": ["call", "call", "put", "put"],
            "strike": [100, 105, 100, 95],
            "marketPrice": [5.0, 2.0, 4.0, 1.5],
            "maturity": [0.1, 0.1, 0.1, 0.1],
            "expiry": pd.to_datetime(["2023-01-01"] * 4),
        }
    )


def test_dashboard_service_initialization(service):
    """
    Tests that the service is initialized with the correct attributes.
    """
    assert service.ticker == "TEST"
    assert service.snapshot_date == "2023-01-01"
    assert service.model_configs is not None
    assert service.final_stock is None
    assert service.final_rate is None


@patch("optpricing.dashboard.service.DailyWorkflow")
def test_run_calibrations(mock_workflow_class, service, mock_market_data):
    """
    Tests that run_calibrations correctly calls the workflow and stores results.
    """
    service._market_data = mock_market_data

    mock_workflow_instance = MagicMock()
    mock_workflow_instance.results = {
        "Model": "BSM",
        "Status": "Success",
        "Calibrated Params": {"sigma": 0.25},
    }

    mock_rate = MagicMock(spec=Rate)
    mock_rate.get_rate.return_value = 0.05
    mock_stock = MagicMock(spec=Stock)
    mock_stock.dividend = 0.01

    mock_workflow_instance.rate = mock_rate
    mock_workflow_instance.stock = mock_stock

    mock_workflow_class.return_value = mock_workflow_instance

    service.run_calibrations()

    mock_workflow_class.assert_called_once()
    config_sent = mock_workflow_class.call_args.kwargs["model_config"]
    assert config_sent["ticker"] == "TEST"

    mock_workflow_instance.run.assert_called_once()
    assert "BSM" in service.calibrated_models
    assert isinstance(service.calibrated_models["BSM"], BSMModel)
    assert service.summary_df.index.name == "Model"
    assert "BSM" in service.summary_df.index


@patch("optpricing.dashboard.service.VolatilitySurface")
def test_calculate_iv_surfaces(mock_vol_surface_class, service, mock_market_data):
    """
    Tests that IV surfaces are calculated correctly after a successful run.
    """
    # Setup a state as if run_calibrations has already succeeded
    service._market_data = mock_market_data
    service.final_rate = Rate(0.05)
    service.final_stock = Stock(100, 0.01)
    service.calibrated_models["BSM"] = BSMModel(params={"sigma": 0.2})

    # Mock the VolatilitySurface object and its chained calls
    mock_surface_instance = MagicMock()
    mock_surface_instance.calculate_market_iv.return_value = mock_surface_instance
    mock_surface_instance.calculate_model_iv.return_value = mock_surface_instance
    mock_surface_instance.surface = pd.DataFrame({"iv": [0.2]})  # Return a dummy df
    mock_vol_surface_class.return_value = mock_surface_instance

    service.calculate_iv_surfaces()

    assert service.market_surface is not None
    assert "BSM" in service.model_surfaces
    assert mock_vol_surface_class.call_count == 2  # Once for market, once for model
    mock_surface_instance.calculate_market_iv.assert_called_once()
    mock_surface_instance.calculate_model_iv.assert_called_once()
