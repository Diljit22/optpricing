from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from optpricing.data import historical_manager


@pytest.fixture
def mock_yfinance_data():
    """Sample DataFrame to be returned by a mocked yfinance."""
    return pd.DataFrame({"Close": [100, 101, 100.5]})


def test_save_historical_returns(monkeypatch, tmp_path, mock_yfinance_data):
    """
    Tests that save_historical_returns correctly calls yfinance and saves a file.
    """
    # Mock yfinance Ticker object
    mock_ticker = MagicMock()
    mock_ticker.history.return_value = mock_yfinance_data
    monkeypatch.setattr("yfinance.Ticker", lambda x: mock_ticker)

    monkeypatch.setattr(historical_manager, "HISTORICAL_DIR", tmp_path)

    historical_manager.save_historical_returns(["TEST"], period="1mo")

    mock_ticker.history.assert_called_once_with(period="1mo")
    expected_file = tmp_path / "TEST_1mo_returns.parquet"
    assert expected_file.exists()


def test_load_historical_returns_existing(monkeypatch, tmp_path):
    """
    Tests loading returns when the file already exists.
    """
    dummy_df = pd.DataFrame({"log_return": [0.01, -0.005]})
    file_path = tmp_path / "TEST_10y_returns.parquet"
    dummy_df.to_parquet(file_path)

    monkeypatch.setattr(historical_manager, "HISTORICAL_DIR", tmp_path)

    returns = historical_manager.load_historical_returns("TEST", period="10y")
    pd.testing.assert_series_equal(returns, dummy_df["log_return"], check_names=False)


@patch("optpricing.data.historical_manager.save_historical_returns")
def test_load_historical_returns_fetches_if_missing(mock_save, tmp_path, monkeypatch):
    """
    Tests that load_historical_returns calls save if the file is missing.
    """
    monkeypatch.setattr(historical_manager, "HISTORICAL_DIR", tmp_path)

    # To simulate failure to download, we just don't create the file after mocking
    with pytest.raises(FileNotFoundError):
        historical_manager.load_historical_returns("TEST", period="10y")

    mock_save.assert_called_once_with(["TEST"], "10y")
