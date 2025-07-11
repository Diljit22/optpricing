from datetime import date
from unittest.mock import MagicMock

import pandas as pd
import pytest
import yfinance as yf

from optpricing.data import market_data_manager


@pytest.fixture
def mock_option_chain_df():
    """Mock option chain dataframe."""
    return pd.DataFrame(
        {
            "strike": [100, 105],
            "bid": [5.0, 2.0],
            "ask": [6.0, 3.0],
            "impliedVolatility": [0.2, 0.18],
            "maturity": [0.1, 0.1],
            "expiry": pd.to_datetime(["2023-12-15", "2023-12-15"]),
            "optionType": ["call", "put"],
            "marketPrice": [5.5, 2.5],
            "spot_price": [100.0, 100.0],
        }
    )


@pytest.fixture
def mock_yfinance_ticker():
    """Mock yfinance Ticker object."""

    class MockOptionChain:
        def __init__(self):
            self.calls = pd.DataFrame(
                {
                    "strike": [100, 105],
                    "bid": [5.0, 2.0],
                    "ask": [6.0, 3.0],
                    "impliedVolatility": [0.2, 0.18],
                }
            )
            self.puts = pd.DataFrame(
                {
                    "strike": [100, 105],
                    "bid": [4.0, 1.5],
                    "ask": [5.0, 2.5],
                    "impliedVolatility": [0.21, 0.19],
                }
            )

    mock_ticker = MagicMock(spec=yf.Ticker)
    mock_ticker.options = ["2023-12-15"]
    mock_ticker.option_chain.return_value = MockOptionChain()
    mock_ticker.fast_info = {"last_price": 100.0}
    mock_ticker.info = {"dividendYield": 2.0}
    return mock_ticker


def test_fetch_from_yfinance_no_expirations(monkeypatch):
    """
    Test _fetch_from_yfinance with no expirations.
    """
    mock_ticker = MagicMock(spec=yf.Ticker)
    mock_ticker.options = []
    monkeypatch.setattr(
        market_data_manager,
        "yf",
        MagicMock(Ticker=lambda ticker: mock_yfinance_ticker),
    )

    result = market_data_manager._fetch_from_yfinance("SPY")
    assert result is None


def test_fetch_from_yfinance_empty_chains(monkeypatch, mock_yfinance_ticker):
    """
    Test _fetch_from_yfinance with empty calls and puts.
    """
    mock_yfinance_ticker.option_chain.return_value = MagicMock(
        calls=pd.DataFrame(), puts=pd.DataFrame()
    )
    monkeypatch.setattr(
        market_data_manager,
        "yf",
        MagicMock(Ticker=lambda ticker: mock_yfinance_ticker),
    )

    result = market_data_manager._fetch_from_yfinance("SPY")
    assert result is None


def test_fetch_from_yfinance_exception(monkeypatch, capsys):
    """
    Test _fetch_from_yfinance with an exception.
    """
    mock_ticker = MagicMock(spec=yf.Ticker)
    mock_ticker.options.side_effect = Exception("API error")
    monkeypatch.setattr(
        market_data_manager,
        "yf",
        MagicMock(Ticker=lambda ticker: mock_yfinance_ticker),
    )

    result = market_data_manager._fetch_from_yfinance("SPY")
    assert result is None
    captured = capsys.readouterr()
    assert "FAILED to fetch live yfinance data for SPY" in captured.out


def test_save_market_snapshot(monkeypatch, tmp_path, mock_option_chain_df):
    """
    Tests that save_market_snapshot calls the live fetcher and saves a file.
    """
    # Mock the live fetcher to return our sample data
    monkeypatch.setattr(
        market_data_manager,
        "get_live_option_chain",
        lambda ticker: mock_option_chain_df,
    )
    # Mock the config directory
    monkeypatch.setattr(market_data_manager, "MARKET_SNAPSHOT_DIR", tmp_path)

    market_data_manager.save_market_snapshot(["TEST"])

    today_str = date.today().strftime("%Y-%m-%d")
    expected_file = tmp_path / f"TEST_{today_str}.parquet"
    assert expected_file.exists()


def test_load_market_snapshot_existing(monkeypatch, tmp_path, mock_option_chain_df):
    """
    Tests loading a snapshot when the file exists.
    """
    file_path = tmp_path / "TEST_2023-01-01.parquet"
    mock_option_chain_df.to_parquet(file_path)
    monkeypatch.setattr(market_data_manager, "MARKET_SNAPSHOT_DIR", tmp_path)

    df = market_data_manager.load_market_snapshot("TEST", "2023-01-01")
    pd.testing.assert_frame_equal(df, mock_option_chain_df)


def test_load_market_snapshot_not_found(monkeypatch, tmp_path):
    """
    Tests that loading a non-existent snapshot returns None.
    """
    monkeypatch.setattr(market_data_manager, "MARKET_SNAPSHOT_DIR", tmp_path)
    df = market_data_manager.load_market_snapshot("TEST", "2023-01-01")
    assert df is None


def test_get_available_snapshot_dates(monkeypatch, tmp_path):
    """
    Tests that available dates are listed and sorted correctly.
    """
    # Create some dummy files
    (tmp_path / "TEST_2023-01-10.parquet").touch()
    (tmp_path / "TEST_2023-01-01.parquet").touch()
    (tmp_path / "OTHER_2023-01-05.parquet").touch()  # Should be ignored

    monkeypatch.setattr(market_data_manager, "MARKET_SNAPSHOT_DIR", tmp_path)

    dates = market_data_manager.get_available_snapshot_dates("TEST")
    assert dates == ["2023-01-10", "2023-01-01"]


def test_save_market_snapshot_empty_data(monkeypatch, tmp_path, capsys):
    """
    Test save_market_snapshot with empty data.
    """
    monkeypatch.setattr(
        market_data_manager,
        "get_live_option_chain",
        lambda ticker: pd.DataFrame(),
    )
    monkeypatch.setattr(market_data_manager, "MARKET_SNAPSHOT_DIR", tmp_path)

    market_data_manager.save_market_snapshot(["TEST"])

    today_str = date.today().strftime("%Y-%m-%d")
    expected_file = tmp_path / f"TEST_{today_str}.parquet"
    assert not expected_file.exists()
    captured = capsys.readouterr()
    assert "No valid option data found for TEST" in captured.out


def test_get_available_snapshot_dates_no_files(monkeypatch, tmp_path):
    """
    Test get_available_snapshot_dates with no files.
    """
    monkeypatch.setattr(market_data_manager, "MARKET_SNAPSHOT_DIR", tmp_path)
    dates = market_data_manager.get_available_snapshot_dates("TEST")
    assert dates == []
