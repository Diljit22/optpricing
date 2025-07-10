from datetime import date

import pandas as pd
import pytest

from optpricing.data import market_data_manager


@pytest.fixture
def mock_option_chain_df():
    """Mock option chain dataframe."""
    return pd.DataFrame({"strike": [100, 105], "last_price": [5.0, 2.5]})


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
