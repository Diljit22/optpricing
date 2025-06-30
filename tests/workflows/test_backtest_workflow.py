from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from optpricing.workflows import BacktestWorkflow


@pytest.fixture
def setup():
    """Provides a standard setup for backtest workflow tests."""
    model_config = {
        "name": "TestModel",
        "model_class": MagicMock(),
    }
    workflow = BacktestWorkflow(ticker="TEST", model_config=model_config)
    return workflow


@patch("optpricing.workflows.backtest_workflow.load_market_snapshot")
@patch("optpricing.workflows.backtest_workflow.get_available_snapshot_dates")
@patch("optpricing.workflows.backtest_workflow.DailyWorkflow")
def test_backtest_workflow_run(
    mock_daily_workflow,
    mock_get_dates,
    mock_load_data,
    setup,
):
    """
    Tests the main run loop of the backtest workflow.
    """
    workflow = setup

    # Mock data layer
    mock_get_dates.return_value = ["2023-01-02", "2023-01-01"]  # Sorted descending
    mock_load_data.return_value = pd.DataFrame()  # Dummy data

    # Mock DailyWorkflow to simulate a successful calibration
    mock_daily_instance = MagicMock()
    mock_daily_instance.results = {
        "Status": "Success",
        "Calibrated Params": {"sigma": 0.25},
    }
    mock_daily_workflow.return_value = mock_daily_instance

    with patch.object(mock_daily_instance, "_evaluate_rmse", return_value=0.5):
        workflow.run()

    assert mock_get_dates.call_count == 1
    assert mock_load_data.call_count == 2  # Once for calib, once for eval
    assert mock_daily_workflow.call_count == 2  # Once for calib, once for eval

    assert len(workflow.results) == 1
    result = workflow.results[0]
    assert result["Eval Date"] == "2023-01-01"
    assert result["Model"] == "TestModel"
    assert result["Out-of-Sample RMSE"] == 0.5


@patch("optpricing.workflows.backtest_workflow.pd.DataFrame.to_csv")
def test_save_results(mock_to_csv, setup):
    """
    Tests that save_results correctly calls the DataFrame's to_csv method.
    """
    workflow = setup

    workflow.results = [
        {"Eval Date": "2023-01-01", "Model": "Test", "Out-of-Sample RMSE": 0.5}
    ]

    workflow.save_results()

    mock_to_csv.assert_called_once()
