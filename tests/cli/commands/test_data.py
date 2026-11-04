from pathlib import Path
from unittest.mock import patch

import pandas as pd
from typer.testing import CliRunner

from optpricing.cli.main import app

runner = CliRunner()


@patch("optpricing.cli.commands.data.save_historical_returns")
def test_data_download_with_tickers(mock_save_hist):
    result = runner.invoke(
        app,
        ["data", "download", "--ticker", "SPY", "--ticker", "AAPL"],
    )
    assert result.exit_code == 0
    mock_save_hist.assert_called_once_with(["SPY", "AAPL"], period="10y")


@patch("optpricing.cli.commands.data.save_historical_returns")
@patch("optpricing.cli.commands.data._config", {"default_tickers": ["TSLA", "GOOGL"]})
def test_data_download_with_all_flag(mock_save_hist):
    result = runner.invoke(app, ["data", "download", "--all"])
    assert result.exit_code == 0
    mock_save_hist.assert_called_once_with(["TSLA", "GOOGL"], period="10y")


@patch("optpricing.cli.commands.data.save_historical_returns")
@patch("optpricing.cli.commands.data.save_risk_free_history")
def test_data_download_rates(mock_rates, mock_save_hist):
    only_rates = runner.invoke(app, ["data", "download", "--rates", "--period", "5y"])
    assert only_rates.exit_code == 0
    mock_rates.assert_called_once_with(period="5y")
    mock_save_hist.assert_not_called()

    both = runner.invoke(app, ["data", "download", "--rates", "-t", "SPY"])
    assert both.exit_code == 0
    mock_save_hist.assert_called_once_with(["SPY"], period="10y")


def test_data_download_requires_target():
    result = runner.invoke(app, ["data", "download"])
    assert result.exit_code == 1
    assert "at least one --ticker" in result.output


@patch("optpricing.cli.commands.data.save_market_snapshot")
def test_data_snapshot_command(mock_save_snapshot):
    mock_save_snapshot.return_value = [Path("data/NVDA_2025-01-01.parquet")]
    result = runner.invoke(app, ["data", "snapshot", "--ticker", "NVDA"])
    assert result.exit_code == 0
    mock_save_snapshot.assert_called_once_with(
        ["NVDA"], compact=False, only_fresh=False
    )
    assert "NVDA_2025-01-01.parquet" in result.output


@patch("optpricing.cli.commands.data.save_market_snapshot", return_value=[])
@patch("optpricing.cli.commands.data._config", {"default_tickers": ["SPY"]})
def test_data_snapshot_compact_nothing_written(mock_save_snapshot):
    result = runner.invoke(
        app, ["data", "snapshot", "--all", "--compact", "--only-fresh"]
    )
    assert result.exit_code == 1
    mock_save_snapshot.assert_called_once_with(["SPY"], compact=True, only_fresh=True)
    assert "No snapshot was written" in result.output


@patch("optpricing.cli.commands.data.record_live_quotes")
@patch(
    "optpricing.cli.commands.data.option_contract_symbols",
    return_value=["SPY250919C00600000"],
)
def test_data_stream_command(mock_contracts, mock_record, tmp_path):
    out = tmp_path / "q.parquet"
    mock_record.return_value = (out, 42)
    result = runner.invoke(
        app,
        [
            "data",
            "stream",
            "-t",
            "spy",
            "-s",
            "^SPX",
            "-d",
            "5",
            "-c",
            "1",
            "-o",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    mock_contracts.assert_called_once_with("spy", 1)
    mock_record.assert_called_once_with(["SPY", "^SPX", "SPY250919C00600000"], 5.0, out)
    assert "Recorded 42 quotes" in result.output


@patch("optpricing.cli.commands.data.record_live_quotes", return_value=(None, 0))
def test_data_stream_no_quotes(mock_record):
    result = runner.invoke(app, ["data", "stream", "-t", "SPY", "-d", "1"])
    assert result.exit_code == 1
    assert "No quotes received" in result.output
    assert runner.invoke(app, ["data", "stream", "-d", "1"]).exit_code == 1


@patch("optpricing.cli.commands.data.get_live_dividend_yield", return_value=0.015)
def test_data_dividends_command(mock_get_yield):
    result = runner.invoke(app, ["data", "dividends", "--ticker", "MSFT"])
    assert result.exit_code == 0
    assert "MSFT" in result.stdout
    assert "1.5000%" in result.stdout
    mock_get_yield.assert_called_once_with("MSFT")


def test_help_lists_every_command():
    result = runner.invoke(app, ["--help"])
    for cmd in ("calibrate", "backtest", "tune", "price", "dashboard", "data", "tools"):
        assert cmd in result.output
    data_help = runner.invoke(app, ["data", "--help"])
    for cmd in ("download", "snapshot", "stream", "dividends"):
        assert cmd in data_help.output


def test_pandas_is_importable_for_tests():
    assert pd is not None
