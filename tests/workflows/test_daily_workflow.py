from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from optpricing.workflows import DailyWorkflow


@pytest.fixture
def setup():
    """Provides a standard setup for daily workflow tests."""
    market_data = pd.DataFrame(
        {
            "spot_price": [100.0],
            "optionType": ["call"],
            "strike": [100],
            "maturity": [0.1],
            "expiry": [pd.Timestamp("2025-01-01")],
            "marketPrice": [10.0],
        }
    )

    model_config = {
        "name": "TestModel",
        "model_class": MagicMock(),
        "initial_guess": {"sigma": 0.2},
        "bounds": {"sigma": (0.01, 1.0)},
    }

    workflow = DailyWorkflow(market_data, model_config)
    return workflow


@patch(
    "optpricing.workflows.daily_workflow.fit_rate_and_dividend",
    return_value=(0.05, 0.01),
)
@patch("optpricing.workflows.daily_workflow.Calibrator")
def test_daily_workflow_success_path(mock_calibrator, mock_fit_r_q, setup):
    """
    Tests the successful execution path of the daily workflow.
    """
    workflow = setup

    # Configure the mock Calibrator's fit method to return a successful result
    mock_calibrator_instance = mock_calibrator.return_value
    mock_calibrator_instance.fit.return_value = {"sigma": 0.25}  # Calibrated param

    # Mock the evaluation to return a fixed RMSE
    with patch.object(workflow, "_evaluate_rmse", return_value=0.5) as mock_evaluate:
        workflow.run()

    mock_fit_r_q.assert_called_once()
    mock_calibrator_instance.fit.assert_called_once()
    mock_evaluate.assert_called_once()

    assert workflow.results["Status"] == "Success"
    assert workflow.results["RMSE"] == 0.5
    assert workflow.results["Calibrated Params"]["sigma"] == 0.25


@patch("optpricing.workflows.daily_workflow.fit_rate_and_dividend")
def test_daily_workflow_failure_path(mock_fit_r_q, setup):
    """
    Tests that the workflow correctly handles exceptions and records a failure.
    """
    workflow = setup

    mock_fit_r_q.side_effect = ValueError("Test Error")

    workflow.run()

    assert workflow.results["Status"] == "Failed"
    assert "RMSE" in workflow.results
    assert "Error" in workflow.results
    assert workflow.results["Error"] == "Test Error"
