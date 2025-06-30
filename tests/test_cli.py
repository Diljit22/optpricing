from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from optpricing.cli import app

# Create a runner to invoke the CLI commands
runner = CliRunner()


@patch("optpricing.cli.subprocess.run")
def test_dashboard_command(mock_subprocess_run):
    """
    Tests that the 'dashboard' command calls streamlit correctly.
    """
    result = runner.invoke(app, ["dashboard"])
    assert result.exit_code == 0
    mock_subprocess_run.assert_called_once()
    assert "streamlit" in mock_subprocess_run.call_args.args[0]
    assert "app.py" in str(mock_subprocess_run.call_args.args[0][-1])


@patch("optpricing.cli.save_historical_returns")
def test_data_download_with_tickers(mock_save_hist):
    """
    Tests the 'data download' subcommand with specific tickers.
    """
    result = runner.invoke(
        app,
        ["data", "download", "--ticker", "SPY", "--ticker", "AAPL"],
    )
    assert result.exit_code == 0
    mock_save_hist.assert_called_once_with(["SPY", "AAPL"], period="10y")


@patch("optpricing.cli.save_historical_returns")
@patch("optpricing.cli._config", {"default_tickers": ["TSLA", "GOOGL"]})
def test_data_download_with_all_flag(mock_save_hist):
    """
    Tests the 'data download' subcommand with the --all flag.
    """
    result = runner.invoke(app, ["data", "download", "--all"])
    assert result.exit_code == 0
    mock_save_hist.assert_called_once_with(["TSLA", "GOOGL"], period="10y")


@patch("optpricing.cli.BacktestWorkflow")
def test_backtest_command(mock_backtest_workflow):
    """
    Tests the 'backtest' command's workflow dispatch.
    """
    mock_workflow_instance = mock_backtest_workflow.return_value

    result = runner.invoke(app, ["backtest", "--ticker", "SPY", "--model", "Heston"])

    assert result.exit_code == 0
    mock_backtest_workflow.assert_called_once()
    mock_workflow_instance.run.assert_called_once()
    mock_workflow_instance.save_results.assert_called_once()
