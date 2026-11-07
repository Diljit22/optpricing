import logging
from datetime import UTC, date, datetime
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import yfinance as yf

from optpricing.atoms import Rate, Stock
from optpricing.data import market_data_manager


@pytest.fixture
def mock_option_chain_df():
    """Mock option chain dataframe (schema v1)."""
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


def make_parity_chain(spot=100.0, r=0.04, q=0.015, maturities=(0.25, 0.5, 1.0)):
    """A synthetic v1 chain whose mids satisfy European put-call parity exactly."""
    rows = []
    for T in maturities:
        F = spot * np.exp((r - q) * T)
        df = np.exp(-r * T)
        for K in np.arange(90, 111, 2.5):
            call = df * max(F - K, 0) + 3.0  # any common time value cancels in C-P
            put = call - df * (F - K)
            for typ, px in (("call", call), ("put", put)):
                rows.append(
                    {
                        "contractSymbol": f"X{T}{K}{typ[0]}",
                        "strike": float(K),
                        "maturity": T,
                        "expiry": pd.Timestamp("2025-01-01"),
                        "optionType": typ,
                        "bid": px - 0.05,
                        "ask": px + 0.05,
                        "marketPrice": px,
                        "volume": 10.0,
                        "openInterest": 100.0,
                        "impliedVolatility": 0.2,
                        "spot_price": spot,
                        "lastTradeDate": pd.Timestamp("2025-01-01 15:00", tz="UTC"),
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture
def mock_yfinance_ticker():
    """Mock yfinance Ticker object."""

    class MockOptionChain:
        def __init__(self):
            self.calls = pd.DataFrame(
                {
                    "contractSymbol": ["C100", "C105"],
                    "strike": [100, 105],
                    "bid": [5.0, 2.0],
                    "ask": [6.0, 3.0],
                    "impliedVolatility": [0.2, 0.18],
                }
            )
            self.puts = pd.DataFrame(
                {
                    "contractSymbol": ["P100", "P105"],
                    "strike": [100, 105],
                    "bid": [4.0, 1.5],
                    "ask": [5.0, 2.5],
                    "impliedVolatility": [0.21, 0.19],
                }
            )

    mock_ticker = MagicMock(spec=yf.Ticker)
    mock_ticker.options = ["2099-12-15"]
    mock_ticker.option_chain.return_value = MockOptionChain()
    mock_ticker.fast_info = {"lastPrice": 100.0}
    mock_ticker.info = {"dividendYield": 2.0}
    return mock_ticker


def test_fetch_from_yfinance_no_expirations(monkeypatch):
    mock_ticker = MagicMock(spec=yf.Ticker)
    mock_ticker.options = []
    monkeypatch.setattr(
        market_data_manager, "yf", MagicMock(Ticker=lambda ticker: mock_ticker)
    )
    assert market_data_manager._fetch_from_yfinance("SPY") is None


def test_fetch_from_yfinance_empty_chains(monkeypatch, mock_yfinance_ticker):
    mock_yfinance_ticker.option_chain.return_value = MagicMock(
        calls=pd.DataFrame(), puts=pd.DataFrame()
    )
    monkeypatch.setattr(
        market_data_manager,
        "yf",
        MagicMock(Ticker=lambda ticker: mock_yfinance_ticker),
    )
    assert market_data_manager._fetch_from_yfinance("SPY") is None


def test_fetch_from_yfinance_builds_v2_schema(monkeypatch, mock_yfinance_ticker):
    """The live fetch attaches ticker, date, rate and implied dividend columns."""
    monkeypatch.setattr(
        market_data_manager,
        "yf",
        MagicMock(Ticker=lambda ticker: mock_yfinance_ticker),
    )
    monkeypatch.setattr(market_data_manager, "get_live_risk_free_rate", lambda: 0.045)

    df = market_data_manager._fetch_from_yfinance("SPY")

    assert market_data_manager.is_v2_snapshot(df)
    assert set(df["optionType"]) == {"call", "put"}
    assert (df["ticker"] == "SPY").all()
    assert df["snapshot_date"].iloc[0] == market_data_manager.session_date().isoformat()
    assert (df["price_source"] == "mid").all()
    assert df["risk_free_rate"].iloc[0] == pytest.approx(0.045)
    assert (df["spot_price"] == 100.0).all()  # fast_info["lastPrice"] was used
    mock_yfinance_ticker.history.assert_not_called()
    # yfinance reports 2.0 meaning 2 %: normalised to a fraction
    assert df["yf_dividend_yield"].iloc[0] == pytest.approx(0.02)


def test_fetch_from_yfinance_exception(monkeypatch, caplog):
    mock_ticker = MagicMock(spec=yf.Ticker)
    type(mock_ticker).options = property(
        lambda self: (_ for _ in ()).throw(Exception("API error"))
    )
    monkeypatch.setattr(
        market_data_manager, "yf", MagicMock(Ticker=lambda ticker: mock_ticker)
    )
    with caplog.at_level(logging.ERROR):
        result = market_data_manager._fetch_from_yfinance("SPY")
    assert result is None
    assert "FAILED to fetch live yfinance data for SPY" in caplog.text


@pytest.mark.parametrize(
    "raw, expected",
    [(None, 0.0), (0.012, 0.012), (1.2, 0.012), (-1.0, 0.0), (float("nan"), 0.0)],
)
def test_normalise_dividend_yield_handles_percent_and_fraction(raw, expected):
    assert market_data_manager._normalise_dividend_yield(raw) == pytest.approx(expected)


def test_save_market_snapshot(monkeypatch, tmp_path, mock_option_chain_df):
    monkeypatch.setattr(
        market_data_manager,
        "get_live_option_chain",
        lambda ticker: mock_option_chain_df,
    )
    monkeypatch.setattr(market_data_manager, "MARKET_SNAPSHOT_DIR", tmp_path)

    written = market_data_manager.save_market_snapshot(["TEST"])

    session = market_data_manager.session_date().isoformat()
    expected_file = tmp_path / f"TEST_{session}.parquet"
    assert expected_file.exists()
    assert written == [expected_file]


def test_save_market_snapshot_compact_and_only_fresh(monkeypatch, tmp_path):
    chain = make_parity_chain()
    chain.loc[0, "bid"] = 0.0  # illiquid row dropped by the compact filter
    monkeypatch.setattr(
        market_data_manager, "get_live_option_chain", lambda ticker: chain
    )
    monkeypatch.setattr(market_data_manager, "MARKET_SNAPSHOT_DIR", tmp_path)

    # Stale chain (last trade in 2025) is skipped when only_fresh is requested.
    assert market_data_manager.save_market_snapshot(["T"], only_fresh=True) == []

    written = market_data_manager.save_market_snapshot(["T"], compact=True)
    assert len(written) == 1
    saved = pd.read_parquet(written[0])
    assert len(saved) == len(chain) - 1
    for dropped in ("contractSymbol", "marketPrice", "lastTradeDate"):
        assert dropped not in saved.columns
    assert saved["bid"].dtype == np.float32
    assert saved["openInterest"].dtype == np.int32
    # the loader restores the mid and the OCC symbol and widens the dtypes
    monkeypatch.setattr(market_data_manager, "MARKET_SNAPSHOT_DIR", tmp_path)
    loaded = market_data_manager.load_market_snapshot(
        "T", written[0].stem.split("_")[1]
    )
    assert loaded["bid"].dtype == np.float64
    np.testing.assert_allclose(
        loaded["marketPrice"], (loaded["bid"] + loaded["ask"]) / 2, rtol=1e-6
    )
    assert loaded["contractSymbol"].iloc[0] == market_data_manager.occ_contract_symbol(
        "T",
        loaded["expiry"].iloc[0],
        loaded["optionType"].iloc[0],
        loaded["strike"].iloc[0],
    )
    stock, rate = market_data_manager.snapshot_market_params(loaded)
    assert (
        rate.get_rate() == pytest.approx(chain["risk_free_rate"].iloc[0])
        if "risk_free_rate" in chain
        else True
    )


def test_occ_contract_symbol():
    assert (
        market_data_manager.occ_contract_symbol(
            "SPY", pd.Timestamp("2025-06-18"), "call", 500.0
        )
        == "SPY250618C00500000"
    )
    assert (
        market_data_manager.occ_contract_symbol(
            "aapl", pd.Timestamp("2026-01-16"), "put", 202.5
        )
        == "AAPL260116P00202500"
    )


def test_liquid_option_mask_rules():
    chain = make_parity_chain(maturities=(0.5,))
    chain.loc[0, "bid"] = 0.0
    chain.loc[1, "openInterest"] = 0.0
    chain.loc[1, "volume"] = 0.0
    chain.loc[2, "ask"] = chain.loc[2, "bid"] * 3  # spread too wide
    chain.loc[3, "maturity"] = 1 / 365.25  # too short
    chain.loc[4, "strike"] = 150.0  # moneyness 1.5 > 1.3
    mask = market_data_manager.liquid_option_mask(chain)
    assert not mask.iloc[:5].any()
    assert mask.iloc[5:].all()
    # volume alone keeps a zero-open-interest row
    chain.loc[1, "volume"] = 1.0
    assert market_data_manager.liquid_option_mask(chain).iloc[1]


def test_load_market_snapshot_upgrades_v1_offline(monkeypatch, tmp_path):
    """A v1 file comes back with the v2 columns, the stored rate and a
    parity-implied dividend yield, without touching the network."""
    chain = make_parity_chain(spot=100.0, r=0.04, q=0.015)
    file_path = tmp_path / "TEST_2025-06-16.parquet"
    chain.to_parquet(file_path)
    monkeypatch.setattr(market_data_manager, "MARKET_SNAPSHOT_DIR", tmp_path)
    monkeypatch.setattr(market_data_manager, "load_risk_free_rate", lambda d: 0.04)
    monkeypatch.setattr(
        market_data_manager, "trailing_dividend_yield", lambda t, d: None
    )
    monkeypatch.setattr(
        market_data_manager,
        "yf",
        MagicMock(Ticker=MagicMock(side_effect=AssertionError("network call"))),
    )

    df = market_data_manager.load_market_snapshot("TEST", "2025-06-16")

    assert market_data_manager.is_v2_snapshot(df)
    assert df["snapshot_date"].iloc[0] == "2025-06-16"
    assert df["risk_free_rate"].iloc[0] == pytest.approx(0.04)
    assert df["dividend_yield"].iloc[0] == pytest.approx(0.015, abs=1e-4)
    pd.testing.assert_frame_equal(df[chain.columns], chain)

    stock, rate = market_data_manager.snapshot_market_params(df)
    assert isinstance(stock, Stock) and isinstance(rate, Rate)
    assert stock.dividend == pytest.approx(0.015, abs=1e-4)
    assert rate.get_rate() == pytest.approx(0.04)


def test_load_market_snapshot_v2_passthrough(monkeypatch, tmp_path):
    chain = market_data_manager.upgrade_snapshot(
        make_parity_chain(), "TEST", "2025-06-16", rate=0.03
    )
    chain.to_parquet(tmp_path / "TEST_2025-06-16.parquet")
    monkeypatch.setattr(market_data_manager, "MARKET_SNAPSHOT_DIR", tmp_path)
    df = market_data_manager.load_market_snapshot("TEST", "2025-06-16")
    assert df["risk_free_rate"].iloc[0] == pytest.approx(0.03)


def test_upgrade_snapshot_without_rate_history_uses_default(monkeypatch, caplog):
    monkeypatch.setattr(market_data_manager, "load_risk_free_rate", lambda d: None)
    monkeypatch.setattr(
        market_data_manager, "trailing_dividend_yield", lambda t, d: None
    )
    with caplog.at_level(logging.WARNING):
        df = market_data_manager.upgrade_snapshot(
            make_parity_chain(), "TEST", "2025-06-16"
        )
    assert df["risk_free_rate"].iloc[0] == market_data_manager.DEFAULT_RISK_FREE_RATE
    assert "No risk-free history" in caplog.text


def test_snapshot_market_params_v1_frame_is_offline():
    chain = make_parity_chain(r=market_data_manager.DEFAULT_RISK_FREE_RATE, q=0.01)
    stock, rate = market_data_manager.snapshot_market_params(chain)
    assert rate.get_rate() == market_data_manager.DEFAULT_RISK_FREE_RATE
    assert stock.dividend == pytest.approx(0.01, abs=1e-4)


def test_session_date_is_new_york_not_utc():
    # 00:30 UTC Saturday is still Friday evening in New York, winter and summer.
    assert market_data_manager.session_date(
        datetime(2026, 11, 7, 0, 30, tzinfo=UTC)
    ) == date(2026, 11, 6)
    assert market_data_manager.session_date(
        datetime(2026, 7, 7, 0, 30, tzinfo=UTC)
    ) == date(2026, 7, 6)
    # An on-time 19:45 UTC run is the same calendar day in both seasons.
    assert market_data_manager.session_date(
        datetime(2026, 11, 6, 19, 45, tzinfo=UTC)
    ) == date(2026, 11, 6)
    assert market_data_manager.session_date(
        datetime(2026, 7, 6, 19, 45, tzinfo=UTC)
    ) == date(2026, 7, 6)


def test_apply_quote_prices_uses_last_trade_when_the_book_is_gone():
    session = date(2026, 11, 6)
    chain = pd.DataFrame(
        {
            "bid": [1.0, 0.0, 0.0],
            "ask": [1.2, 0.0, 0.0],
            "lastPrice": [1.05, 2.5, 3.0],
            "lastTradeDate": [
                pd.Timestamp("2026-11-06 15:00", tz="UTC"),
                pd.Timestamp("2026-11-06 21:00", tz="UTC"),  # 16:00 ET
                pd.Timestamp("2026-11-05 21:00", tz="UTC"),  # previous session
            ],
            "strike": [100.0, 101.0, 102.0],
        }
    )
    out = market_data_manager.apply_quote_prices(chain, session)
    assert out["price_source"].tolist() == ["mid", "last"]
    assert out["bid"].iloc[0] == 1.0
    assert out["ask"].iloc[0] == 1.2
    assert out["marketPrice"].iloc[0] == pytest.approx(1.1)
    assert out["bid"].iloc[1] == pytest.approx(2.5)
    assert out["ask"].iloc[1] == pytest.approx(2.5)
    assert out["marketPrice"].iloc[1] == pytest.approx(2.5)
    assert 102.0 not in out["strike"].to_numpy()


def test_compact_keeps_last_trade_marks_for_the_calibrator():
    """A cleared book still produces a compact chain the liquidity filter accepts."""
    session = date(2026, 11, 6)
    chain = make_parity_chain()
    chain["bid"] = 0.0
    chain["ask"] = 0.0
    chain["lastPrice"] = chain["marketPrice"]
    chain["lastTradeDate"] = pd.Timestamp("2026-11-06 21:00", tz="UTC")
    priced = market_data_manager.apply_quote_prices(chain, session)
    compact = market_data_manager.compact_snapshot(priced)
    assert not compact.empty
    assert (compact["price_source"] == "last").all()
    assert (compact["bid"] > 0).all()
    assert "lastPrice" not in compact.columns


def test_save_snapshot_uses_new_york_session_date(monkeypatch, tmp_path):
    """A run after midnight UTC still files the chain under the NY session."""
    chain = make_parity_chain()
    chain["lastTradeDate"] = pd.Timestamp("2026-11-06 23:00", tz="UTC")
    monkeypatch.setattr(
        market_data_manager, "session_date", lambda now=None: date(2026, 11, 6)
    )
    monkeypatch.setattr(
        market_data_manager, "get_live_option_chain", lambda ticker: chain
    )
    monkeypatch.setattr(market_data_manager, "MARKET_SNAPSHOT_DIR", tmp_path)

    written = market_data_manager.save_market_snapshot(["T"], only_fresh=True)

    assert written == [tmp_path / "T_2026-11-06.parquet"]


def test_is_fresh_snapshot():
    chain = make_parity_chain()
    assert not market_data_manager.is_fresh_snapshot(chain, date(2025, 1, 2))
    assert market_data_manager.is_fresh_snapshot(chain, date(2025, 1, 1))
    assert market_data_manager.is_fresh_snapshot(chain.drop(columns="lastTradeDate"))
    assert not market_data_manager.is_fresh_snapshot(chain.iloc[0:0])
    # pre-market: most of the raw chain had no quote and was filtered away
    chain.attrs["n_raw_quotes"] = len(chain) * 20
    assert not market_data_manager.is_fresh_snapshot(chain, date(2025, 1, 1))
    chain.attrs["n_raw_quotes"] = len(chain) + 2
    assert market_data_manager.is_fresh_snapshot(chain, date(2025, 1, 1))


def test_load_market_snapshot_not_found(monkeypatch, tmp_path):
    monkeypatch.setattr(market_data_manager, "MARKET_SNAPSHOT_DIR", tmp_path)
    assert market_data_manager.load_market_snapshot("TEST", "2023-01-01") is None


def test_get_available_snapshot_dates(monkeypatch, tmp_path):
    (tmp_path / "TEST_2023-01-10.parquet").touch()
    (tmp_path / "TEST_2023-01-01.parquet").touch()
    (tmp_path / "OTHER_2023-01-05.parquet").touch()  # Should be ignored
    monkeypatch.setattr(market_data_manager, "MARKET_SNAPSHOT_DIR", tmp_path)

    assert market_data_manager.get_available_snapshot_dates("TEST") == [
        "2023-01-10",
        "2023-01-01",
    ]
    assert market_data_manager.get_available_snapshot_dates("TEST", ascending=True) == [
        "2023-01-01",
        "2023-01-10",
    ]


def test_save_market_snapshot_empty_data(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(
        market_data_manager, "get_live_option_chain", lambda ticker: pd.DataFrame()
    )
    monkeypatch.setattr(market_data_manager, "MARKET_SNAPSHOT_DIR", tmp_path)

    with caplog.at_level(logging.WARNING):
        written = market_data_manager.save_market_snapshot(["TEST"])

    assert written == []
    assert "No valid option data found for TEST" in caplog.text


def test_get_available_snapshot_dates_no_files(monkeypatch, tmp_path):
    monkeypatch.setattr(market_data_manager, "MARKET_SNAPSHOT_DIR", tmp_path)
    assert market_data_manager.get_available_snapshot_dates("TEST") == []


def test_get_live_dividend_yield_normalises_units(monkeypatch):
    ticker = MagicMock()
    ticker.info = {"dividendYield": 1.25}
    monkeypatch.setattr(market_data_manager, "yf", MagicMock(Ticker=lambda t: ticker))
    assert market_data_manager.get_live_dividend_yield("X") == pytest.approx(0.0125)
    ticker.info = {"trailingAnnualDividendYield": 0.02}
    assert market_data_manager.get_live_dividend_yield("X") == pytest.approx(0.02)
