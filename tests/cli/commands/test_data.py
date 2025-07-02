from unittest.mock import patch

from typer.testing import CliRunner

from optpricing.cli.main import app

runner = CliRunner()


@patch("optpricing.cli.commands.data.save_historical_returns")
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


@patch("optpricing.cli.commands.data.save_historical_returns")
@patch("optpricing.cli.commands.data._config", {"default_tickers": ["TSLA", "GOOGL"]})
def test_data_download_with_all_flag(mock_save_hist):
    """
    Tests the 'data download' subcommand with the --all flag.
    """
    result = runner.invoke(app, ["data", "download", "--all"])
    assert result.exit_code == 0
    mock_save_hist.assert_called_once_with(["TSLA", "GOOGL"], period="10y")


@patch("optpricing.cli.commands.data.save_market_snapshot")
def test_data_snapshot_command(mock_save_snapshot):
    """
    Tests the 'data snapshot' subcommand.
    """
    result = runner.invoke(app, ["data", "snapshot", "--ticker", "NVDA"])
    assert result.exit_code == 0
    mock_save_snapshot.assert_called_once_with(["NVDA"])


@patch("optpricing.cli.commands.data.get_live_dividend_yield", return_value=0.015)
def test_data_dividends_command(mock_get_yield):
    """
    Tests the 'data dividends' subcommand.
    """
    result = runner.invoke(app, ["data", "dividends", "--ticker", "MSFT"])
    assert result.exit_code == 0
    assert "MSFT" in result.stdout
    assert "1.5000%" in result.stdout
    mock_get_yield.assert_called_once_with("MSFT")
