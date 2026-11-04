from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from optpricing.atoms import Rate, Stock
from optpricing.calibration import CalibrationSettings
from optpricing.dashboard import service as service_module
from optpricing.dashboard.service import DashboardService
from optpricing.models import BSMModel, SABRJumpModel
from optpricing.workflows.configs import ALL_MODEL_CONFIGS


@pytest.fixture
def mock_model_configs():
    """Sample model configs."""
    return {"BSM": {"model_class": BSMModel, "name": "BSM"}}


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
    assert service.ticker == "TEST"
    assert service.snapshot_date == "2023-01-01"
    assert service.model_configs is not None
    assert service.final_stock is None
    assert service.final_rate is None
    assert isinstance(service.settings, CalibrationSettings)


@patch("optpricing.dashboard.service.DailyWorkflow")
def test_run_calibrations(mock_workflow_class, service, mock_market_data):
    service._market_data = mock_market_data

    mock_workflow_instance = MagicMock()
    mock_workflow_instance.results = {
        "Model": "BSM",
        "Status": "Success",
        "Calibrated Params": {"sigma": 0.25},
        "IV RMSE": 0.01,
        "Buckets": pd.DataFrame({"bucket": ["ATM"], "rmse_iv": [0.01]}),
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
    kwargs = mock_workflow_class.call_args.kwargs
    assert kwargs["model_config"]["ticker"] == "TEST"
    assert isinstance(kwargs["settings"], CalibrationSettings)
    assert kwargs["allow_mc"] is False
    assert "ticker" not in service.model_configs["BSM"]  # the recipe is not mutated

    mock_workflow_instance.run.assert_called_once()
    assert "BSM" in service.calibrated_models
    assert isinstance(service.calibrated_models["BSM"], BSMModel)
    assert service.summary_df.index.name == "Model"
    assert "BSM" in service.summary_df.index
    assert service.summary_df.loc["BSM", "Calibrated Params"] == "sigma=0.25"
    assert "Buckets" not in service.summary_df.columns
    assert "BSM" in service.bucket_tables
    assert service.final_rate is mock_rate


def test_settings_for_prefers_tuned(service, monkeypatch):
    tuned = CalibrationSettings(name="near-the-money")
    monkeypatch.setattr(service_module, "load_tuned_settings", lambda t, m: tuned)
    assert service.settings_for("BSM") is tuned
    service.use_tuned = False
    assert service.settings_for("BSM") is service.settings
    service.use_tuned = True
    monkeypatch.setattr(service_module, "load_tuned_settings", lambda t, m: None)
    assert service.settings_for("BSM") is service.settings


def test_end_to_end_on_synthetic_snapshot(bsm_snapshot, no_network):
    """Real workflow, real surfaces (no Streamlit UI assertions)."""
    configs = {
        "BSM": ALL_MODEL_CONFIGS["BSM"],
        "SABRJump": ALL_MODEL_CONFIGS["SABRJump"],
    }
    service = DashboardService(
        "TEST",
        "2025-06-16",
        configs,
        settings=CalibrationSettings(n_starts=1),
        use_tuned=False,
    )
    service._market_data = bsm_snapshot
    service.run_calibrations()
    assert list(service.summary_df.index) == ["BSM", "SABRJump"]  # success first
    assert service.summary_df.loc["BSM", "Status"] == "Success"
    assert service.summary_df.loc["SABRJump", "Status"] == "Failed"
    assert set(service.calibrated_models) == {"BSM"}
    service.calculate_iv_surfaces()
    assert not service.market_surface.empty
    assert set(service.model_surfaces) == {"BSM"}
    assert service.model_surfaces["BSM"]["iv"].sub(0.23).abs().max() < 1e-3


def test_surface_skips_models_that_cannot_be_priced(bsm_snapshot, caplog):
    service = DashboardService("TEST", "2025-06-16", {}, use_tuned=False)
    service._market_data = bsm_snapshot
    service.final_rate, service.final_stock = Rate(0.04), Stock(100.0, dividend=0.01)
    service.calibrated_models["SABRJump"] = SABRJumpModel()
    service.calculate_iv_surfaces()
    assert "SABRJump" not in service.model_surfaces
    assert "No IV surface for SABRJump" in caplog.text


@patch("optpricing.dashboard.service.VolatilitySurface")
def test_calculate_iv_surfaces(mock_vol_surface_class, service, mock_market_data):
    service._market_data = mock_market_data
    service.final_rate = Rate(0.05)
    service.final_stock = Stock(100, 0.01)
    service.calibrated_models["BSM"] = BSMModel(params={"sigma": 0.2})

    mock_surface_instance = MagicMock()
    mock_surface_instance.calculate_market_iv.return_value = mock_surface_instance
    mock_surface_instance.calculate_model_iv.return_value = mock_surface_instance
    mock_surface_instance.surface = pd.DataFrame({"iv": [0.2]})
    mock_vol_surface_class.return_value = mock_surface_instance

    service.calculate_iv_surfaces()

    assert service.market_surface is not None
    assert "BSM" in service.model_surfaces
    assert mock_vol_surface_class.call_count == 2
    mock_surface_instance.calculate_market_iv.assert_called_once()
    mock_surface_instance.calculate_model_iv.assert_called_once()
    assert (
        mock_surface_instance.calculate_model_iv.call_args.kwargs["allow_mc"] is False
    )


def test_market_data_errors_when_unavailable(monkeypatch):
    monkeypatch.setattr(service_module, "load_market_snapshot", lambda t, d: None)
    with pytest.raises(ValueError, match="Could not load data"):
        _ = DashboardService("TEST", "2025-06-16", {}).market_data
