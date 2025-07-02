from unittest.mock import patch

from typer.testing import CliRunner

from optpricing.cli.main import app

runner = CliRunner()


@patch("optpricing.cli.commands.backtest.BacktestWorkflow")
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
