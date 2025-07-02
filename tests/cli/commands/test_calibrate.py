from unittest.mock import patch

import pandas as pd
from typer.testing import CliRunner

from optpricing.cli.main import app

runner = CliRunner()


@patch("optpricing.cli.commands.calibrate.DailyWorkflow")
@patch(
    "optpricing.cli.commands.calibrate.load_market_snapshot",
    return_value=pd.DataFrame([{"spot": 100}]),
)
@patch(
    "optpricing.cli.commands.calibrate.get_available_snapshot_dates",
    return_value=["2023-01-01"],
)
def test_calibrate_command(mock_get_dates, mock_load_data, mock_daily_workflow):
    """
    Tests the 'calibrate' command's workflow dispatch.
    """
    mock_workflow_instance = mock_daily_workflow.return_value
    mock_workflow_instance.results = {
        "Status": "Success",
        "RMSE": 0.01,
        "Calibrated Params": {"sigma": 0.2},
    }

    result = runner.invoke(
        app, ["calibrate", "--ticker", "SPY", "--model", "BSM", "--date", "2023-01-01"]
    )

    assert result.exit_code == 0
    assert "SUCCEEDED" in result.stdout
    mock_load_data.assert_called_once_with("SPY", "2023-01-01")
    mock_daily_workflow.assert_called_once()
    mock_workflow_instance.run.assert_called_once()
