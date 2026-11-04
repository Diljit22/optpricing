import threading
import time
from datetime import UTC, datetime

import pandas as pd
import pytest

from optpricing.data import stream


class FakeWebSocket:
    """Emits a few messages then blocks until closed, like yfinance.WebSocket."""

    def __init__(self, messages):
        self.messages = list(messages)
        self.subscribed = []
        self.closed = threading.Event()
        self.subscribe_calls = 0

    def subscribe(self, symbols):
        self.subscribe_calls += 1
        self.subscribed = list(symbols)

    def listen(self, handler):
        for msg in self.messages:
            handler(msg)
        self.closed.wait(timeout=10)

    def close(self):
        self.closed.set()


SPY_MSG = {
    "id": "SPY",
    "price": 601.25,
    "time": "1750100000000",
    "exchange": "PCX",
    "quote_type": 8,
    "market_hours": 1,
    "change_percent": 0.5,
    "day_volume": "12345678",
    "bid": 601.2,
    "bid_size": "3",
    "ask": 601.3,
    "ask_size": "5",
}
OPT_MSG = {
    "id": "SPY250919C00600000",
    "price": 15.2,
    "time": "1750100001000",
    "underlying_symbol": "SPY",
    "strike_price": 600.0,
    "expire_date": "1758240000",
    "options_type": 0,
    "open_interest": "4321",
    "quote_type": 15,
}


def test_normalise_message_types():
    row = stream.normalise_message(SPY_MSG, datetime(2025, 6, 16, tzinfo=UTC))
    assert row["symbol"] == "SPY"
    assert row["time"] == 1750100000000 and isinstance(row["time"], int)
    assert row["bid_size"] == 3
    assert row["strike_price"] is None
    frame = stream.quotes_to_frame([row])
    assert frame["quote_time"].iloc[0] == pd.Timestamp(1750100000, unit="s", tz="UTC")
    assert str(frame["day_volume"].dtype) == "Int64"


def test_record_live_quotes_writes_parquet(tmp_path):
    ws = FakeWebSocket([SPY_MSG, OPT_MSG, {"error": "bad"}, SPY_MSG])
    out = tmp_path / "rec.parquet"

    path, n = stream.record_live_quotes(
        ["spy", "SPY250919C00600000"],
        duration=0.5,
        output=out,
        websocket_factory=lambda: ws,
        heartbeat=0.1,
    )

    assert path == out and n == 3  # message without id is ignored
    assert ws.subscribed == ["SPY", "SPY250919C00600000"]
    assert ws.subscribe_calls >= 2  # initial + heartbeat re-subscribe
    assert ws.closed.is_set()
    saved = pd.read_parquet(out)
    assert len(saved) == 3
    assert set(saved["symbol"]) == {"SPY", "SPY250919C00600000"}
    assert saved.loc[saved["symbol"] != "SPY", "strike_price"].iloc[0] == 600.0


def test_record_live_quotes_no_messages(tmp_path):
    ws = FakeWebSocket([])
    path, n = stream.record_live_quotes(
        ["SPY"],
        duration=0.2,
        output=tmp_path / "x.parquet",
        websocket_factory=lambda: ws,
    )
    assert path is None and n == 0
    assert not (tmp_path / "x.parquet").exists()


def test_record_live_quotes_requires_symbols():
    with pytest.raises(ValueError):
        stream.record_live_quotes([], duration=0.1)


def test_recorder_flushes_periodically(tmp_path):
    rec = stream.QuoteRecorder(tmp_path / "r.parquet", flush_every=2)
    rec(SPY_MSG)
    assert not (tmp_path / "r.parquet").exists()
    rec(OPT_MSG)
    assert (tmp_path / "r.parquet").exists()
    assert rec.n_messages == 2


def test_default_recording_path(monkeypatch, tmp_path):
    monkeypatch.setattr(stream, "LIVE_QUOTES_DIR", tmp_path)
    when = datetime(2025, 6, 16, 14, 30, 5, tzinfo=UTC)
    assert stream.default_recording_path(["^SPX", "SPY"], when) == (
        tmp_path / "SPX+1_2025-06-16_143005.parquet"
    )


def test_option_contract_symbols_ranks_by_open_interest():
    chain = pd.DataFrame(
        {
            "contractSymbol": ["A", "B", "C", "D"],
            "expiry": pd.to_datetime(["2099-01-01"] * 3 + ["2000-01-01"]),
            "openInterest": [10.0, 500.0, 500.0, 9999.0],
            "volume": [1.0, 2.0, 50.0, 1.0],
        }
    )
    assert stream.option_contract_symbols("X", 2, chain=chain) == ["C", "B"]
    assert stream.option_contract_symbols("X", 0, chain=chain) == []
    assert stream.option_contract_symbols("X", 3, chain=pd.DataFrame()) == []


def test_record_live_quotes_stops_when_listener_dies(tmp_path):
    class DyingWebSocket(FakeWebSocket):
        def listen(self, handler):
            handler(SPY_MSG)  # then returns immediately (connection dropped)

    ws = DyingWebSocket([])
    start = time.monotonic()
    path, n = stream.record_live_quotes(
        ["SPY"],
        duration=30,
        output=tmp_path / "d.parquet",
        websocket_factory=lambda: ws,
    )
    assert n == 1 and path is not None
    assert time.monotonic() - start < 5
