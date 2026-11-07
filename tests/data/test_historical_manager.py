import math
from datetime import date
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from optpricing.data import historical_manager


@pytest.fixture
def mock_yfinance_data():
    """Sample history frame as returned by yfinance (auto-adjusted)."""
    idx = pd.date_range("2024-01-02", periods=4, freq="B", tz="America/New_York")
    return pd.DataFrame(
        {"Close": [100.0, 101.0, 100.5, 102.0], "Dividends": [0.0, 0.5, 0.0, 0.0]},
        index=idx,
    )


def test_save_historical_returns_writes_v2_frame(
    monkeypatch, tmp_path, mock_yfinance_data
):
    mock_ticker = MagicMock()
    mock_ticker.history.return_value = mock_yfinance_data
    monkeypatch.setattr("yfinance.Ticker", lambda x: mock_ticker)
    monkeypatch.setattr(historical_manager, "HISTORICAL_DIR", tmp_path)

    historical_manager.save_historical_returns(["TEST"], period="1mo")

    mock_ticker.history.assert_called_once_with(period="1mo", auto_adjust=True)
    frame = pd.read_parquet(tmp_path / "TEST_1mo_returns.parquet")
    assert list(frame.columns) == ["close", "dividends", "log_return"]
    assert len(frame) == 3  # first return is NaN and dropped
    assert frame["log_return"].iloc[0] == pytest.approx(math.log(101.0 / 100.0))
    assert frame["dividends"].iloc[0] == 0.5


def test_load_historical_returns_existing_v1(monkeypatch, tmp_path):
    """v1 files (single log_return column) still load."""
    dummy_df = pd.DataFrame({"log_return": [0.01, -0.005]})
    dummy_df.to_parquet(tmp_path / "TEST_10y_returns.parquet")
    monkeypatch.setattr(historical_manager, "HISTORICAL_DIR", tmp_path)

    returns = historical_manager.load_historical_returns("TEST", period="10y")
    pd.testing.assert_series_equal(returns, dummy_df["log_return"], check_names=False)
    assert historical_manager.trailing_dividend_yield("TEST", "2024-06-01") is None


@patch("optpricing.data.historical_manager.save_historical_returns")
def test_load_historical_returns_fetches_if_missing(mock_save, tmp_path, monkeypatch):
    monkeypatch.setattr(historical_manager, "HISTORICAL_DIR", tmp_path)
    with pytest.raises(FileNotFoundError):
        historical_manager.load_historical_returns("TEST", period="10y")
    mock_save.assert_called_once_with(["TEST"], "10y")


def test_trailing_dividend_yield(monkeypatch, tmp_path):
    idx = pd.date_range("2024-01-01", periods=400, freq="D")
    frame = pd.DataFrame(
        {"close": 100.0, "dividends": 0.0, "log_return": 0.0}, index=idx
    )
    # Four quarterly dividends of 0.5 inside the trailing year, one just outside.
    for day in ("2024-02-01", "2024-05-01", "2024-08-01", "2024-11-01"):
        frame.loc[pd.Timestamp(day), "dividends"] = 0.5
    frame.loc[pd.Timestamp("2024-01-05"), "dividends"] = 9.0  # > 365 days before
    frame.to_parquet(tmp_path / "TEST_10y_returns.parquet")
    monkeypatch.setattr(historical_manager, "HISTORICAL_DIR", tmp_path)

    q = historical_manager.trailing_dividend_yield("TEST", "2025-01-10")
    assert q == pytest.approx(math.log1p(2.0 / 100.0))
    assert historical_manager.trailing_dividend_yield("TEST", "2023-01-01") is None


def test_tbill_discount_to_continuous():
    # 4.25 % discount yield: price = 1 - 0.0425*91/360, r = -ln(price)*365/91
    r = historical_manager.tbill_discount_to_continuous(4.25)
    assert r == pytest.approx(-math.log(1 - 0.0425 * 91 / 360) * 365 / 91)
    assert 0.042 < r < 0.045
    with pytest.raises(ValueError):
        historical_manager.tbill_discount_to_continuous(500.0)


def test_risk_free_history_roundtrip(monkeypatch, tmp_path):
    idx = pd.date_range("2025-06-10", periods=5, freq="B", tz="America/New_York")
    hist = pd.DataFrame({"Close": [4.2, 4.21, np.nan, 4.19, 4.3]}, index=idx)
    mock_ticker = MagicMock()
    mock_ticker.history.return_value = hist
    monkeypatch.setattr("yfinance.Ticker", lambda x: mock_ticker)
    monkeypatch.setattr(historical_manager, "HISTORICAL_DIR", tmp_path)

    historical_manager.save_risk_free_history(period="1mo")

    frame = pd.read_parquet(tmp_path / historical_manager.RATE_FILE)
    assert len(frame) == 4  # NaN dropped
    assert frame.index.tz is None
    # Exact date, a weekend (previous observation) and a too-old request.
    r = historical_manager.load_risk_free_rate("2025-06-11")
    assert r == pytest.approx(historical_manager.tbill_discount_to_continuous(4.21))
    assert historical_manager.load_risk_free_rate("2025-06-15") == pytest.approx(
        historical_manager.tbill_discount_to_continuous(4.19)
    )
    assert historical_manager.load_risk_free_rate("2025-09-01") is None
    assert historical_manager.load_risk_free_rate("2025-06-01") is None


def test_load_risk_free_rate_without_file(monkeypatch, tmp_path):
    monkeypatch.setattr(historical_manager, "HISTORICAL_DIR", tmp_path)
    assert historical_manager.load_risk_free_rate("2025-06-11") is None


def test_history_covers_session(monkeypatch, tmp_path):
    idx = pd.to_datetime(["2026-11-05", "2026-11-06"]).tz_localize("America/New_York")
    pd.DataFrame({"close": [1.0, 1.0]}, index=idx).to_parquet(
        tmp_path / "SPY_10y_returns.parquet"
    )
    monkeypatch.setattr(historical_manager, "HISTORICAL_DIR", tmp_path)
    assert historical_manager.history_covers_session(date(2026, 11, 6))
    assert not historical_manager.history_covers_session(date(2026, 11, 7))
    assert not historical_manager.history_covers_session(date(2026, 11, 6), period="5y")


def test_get_live_risk_free_rate(monkeypatch):
    ticker = MagicMock()
    ticker.fast_info = {"lastPrice": 4.0}
    monkeypatch.setattr("yfinance.Ticker", lambda x: ticker)
    assert historical_manager.get_live_risk_free_rate() == pytest.approx(
        historical_manager.tbill_discount_to_continuous(4.0)
    )
    ticker.fast_info = {}
    assert historical_manager.get_live_risk_free_rate() is None
