from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Iterable
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from optpricing.config import LIVE_QUOTES_DIR

__doc__ = """
Record live quotes from Yahoo Finance's free streaming endpoint
(``yfinance.WebSocket``) into parquet files.

Each decoded message is one row. Underlyings and option contracts (e.g.
``SPY250919C00600000``) can be streamed together; option messages carry
``strike_price``, ``expire_date``, ``options_type`` and ``open_interest`` so a
recording can later be turned into an intraday snapshot.
"""

logger = logging.getLogger(__name__)

# Fields of yfinance's PricingData protobuf that are kept (name -> dtype).
QUOTE_FIELDS: dict[str, str] = {
    "id": "string",
    "time": "int64",
    "price": "float64",
    "bid": "float64",
    "ask": "float64",
    "bid_size": "int64",
    "ask_size": "int64",
    "last_size": "int64",
    "day_volume": "int64",
    "day_high": "float64",
    "day_low": "float64",
    "open_price": "float64",
    "previous_close": "float64",
    "change": "float64",
    "change_percent": "float64",
    "market_hours": "int64",
    "quote_type": "int64",
    "exchange": "string",
    "currency": "string",
    "underlying_symbol": "string",
    "strike_price": "float64",
    "expire_date": "int64",
    "options_type": "int64",
    "open_interest": "int64",
}


def normalise_message(message: dict[str, Any], received_at: datetime) -> dict[str, Any]:
    """
    Flattens one decoded streamer message into a typed row.

    Parameters
    ----------
    message : dict[str, Any]
        Message as produced by ``yfinance.WebSocket`` (``MessageToDict`` with
        protobuf field names; 64-bit integers arrive as strings).
    received_at : datetime
        Local receipt time (UTC).

    Returns
    -------
    dict[str, Any]
        Row with every key of :data:`QUOTE_FIELDS` plus ``received_at`` and
        ``symbol`` (alias of ``id``).
    """
    row: dict[str, Any] = {}
    for field, dtype in QUOTE_FIELDS.items():
        value = message.get(field)
        if value is None:
            row[field] = None
        elif dtype == "int64":
            try:
                row[field] = int(value)
            except (TypeError, ValueError):
                row[field] = None
        elif dtype == "float64":
            try:
                row[field] = float(value)
            except (TypeError, ValueError):
                row[field] = None
        else:
            row[field] = str(value)
    row["symbol"] = row["id"]
    row["received_at"] = received_at
    return row


def quotes_to_frame(rows: Iterable[dict[str, Any]]) -> pd.DataFrame:
    """
    Builds a typed DataFrame from normalised rows.

    ``time`` (epoch milliseconds) is converted to a UTC timestamp column
    ``quote_time``.
    """
    frame = pd.DataFrame(list(rows))
    if frame.empty:
        return frame
    for field, dtype in QUOTE_FIELDS.items():
        if field not in frame.columns:
            continue
        if dtype == "int64":
            frame[field] = frame[field].astype("Int64")
        elif dtype == "float64":
            frame[field] = pd.to_numeric(frame[field], errors="coerce")
        else:
            frame[field] = frame[field].astype("string")
    frame["quote_time"] = pd.to_datetime(frame["time"], unit="ms", utc=True)
    frame["received_at"] = pd.to_datetime(frame["received_at"], utc=True)
    return frame


def default_recording_path(symbols: list[str], when: datetime | None = None) -> Path:
    """``data/live_quotes/{first-symbol}[+N]_{YYYY-MM-DD}_{HHMMSS}.parquet``."""
    when = when or datetime.now(UTC)
    head = symbols[0].replace("^", "").upper()
    suffix = f"+{len(symbols) - 1}" if len(symbols) > 1 else ""
    stamp = when.strftime("%Y-%m-%d_%H%M%S")
    return LIVE_QUOTES_DIR / f"{head}{suffix}_{stamp}.parquet"


class QuoteRecorder:
    """
    Collects streamer messages in memory and writes them to parquet.

    Parameters
    ----------
    output : Path
        Destination parquet file (zstd-compressed).
    flush_every : int, optional
        Rewrite the file every this many messages so a long recording survives
        an interruption, by default 1000.
    """

    def __init__(self, output: Path, *, flush_every: int = 1000):
        self.output = Path(output)
        self.flush_every = max(int(flush_every), 1)
        self._rows: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._since_flush = 0

    def __call__(self, message: dict[str, Any]) -> None:
        """Message handler for ``WebSocket.listen``."""
        if not isinstance(message, dict) or "id" not in message:
            return
        row = normalise_message(message, datetime.now(UTC))
        with self._lock:
            self._rows.append(row)
            self._since_flush += 1
            flush = self._since_flush >= self.flush_every
            if flush:
                self._since_flush = 0
        if flush:
            self.flush()

    @property
    def n_messages(self) -> int:
        with self._lock:
            return len(self._rows)

    def frame(self) -> pd.DataFrame:
        with self._lock:
            rows = list(self._rows)
        return quotes_to_frame(rows)

    def flush(self) -> Path | None:
        """Writes everything collected so far; returns the path (None if empty)."""
        frame = self.frame()
        if frame.empty:
            return None
        self.output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(self.output, compression="zstd", index=False)
        return self.output


def record_live_quotes(
    symbols: list[str],
    duration: float,
    output: Path | str | None = None,
    *,
    websocket_factory: Callable[[], Any] | None = None,
    heartbeat: float = 15.0,
    flush_every: int = 1000,
) -> tuple[Path | None, int]:
    """
    Streams quotes for ``symbols`` for ``duration`` seconds and saves them.

    Parameters
    ----------
    symbols : list[str]
        Yahoo symbols (tickers, indices such as ``^SPX`` or option contract
        symbols).
    duration : float
        Recording length in seconds (Ctrl-C stops early and still saves).
    output : Path | str | None, optional
        Parquet destination; defaults to :func:`default_recording_path`.
    websocket_factory : Callable[[], Any] | None, optional
        Returns an object with ``subscribe``, ``listen(handler)`` and ``close``
        (defaults to ``yfinance.WebSocket(verbose=False)``); injectable for tests.
    heartbeat : float, optional
        Seconds between re-subscriptions keeping the stream alive, by default 15.
    flush_every : int, optional
        Messages between intermediate writes, by default 1000.

    Returns
    -------
    tuple[Path | None, int]
        The written file (None when no message arrived) and the message count.
    """
    if not symbols:
        raise ValueError("At least one symbol is required.")
    symbols = [s.upper() for s in symbols]
    path = Path(output) if output is not None else default_recording_path(symbols)
    recorder = QuoteRecorder(path, flush_every=flush_every)

    if websocket_factory is None:
        import warnings

        import yfinance as yf

        # yfinance 1.x still uses the non-context-manager `websockets` API.
        warnings.filterwarnings(
            "ignore", category=DeprecationWarning, module=r"yfinance\.live"
        )

        def websocket_factory():
            return yf.WebSocket(verbose=False)

    ws = websocket_factory()
    ws.subscribe(symbols)
    listener = threading.Thread(target=ws.listen, args=(recorder,), daemon=True)
    listener.start()
    logger.info("Streaming %s for %.0f s -> %s", symbols, duration, path)

    deadline = time.monotonic() + float(duration)
    next_heartbeat = time.monotonic() + heartbeat
    try:
        while time.monotonic() < deadline and listener.is_alive():
            time.sleep(min(0.25, max(deadline - time.monotonic(), 0.0)))
            if heartbeat > 0 and time.monotonic() >= next_heartbeat:
                try:
                    ws.subscribe(symbols)
                except Exception as e:  # pragma: no cover - network specific
                    logger.warning("Heartbeat re-subscribe failed: %s", e)
                next_heartbeat = time.monotonic() + heartbeat
    except KeyboardInterrupt:
        logger.info("Interrupted; saving what was received.")
    finally:
        # Closing the socket makes the blocked `recv` in the listener thread
        # raise ConnectionClosedOK, which yfinance logs with a traceback; that
        # is the expected shutdown path, so silence its logger meanwhile.
        yf_logger = logging.getLogger("yfinance")
        previous_level = yf_logger.level
        yf_logger.setLevel(logging.CRITICAL)
        try:
            ws.close()
        except Exception:  # pragma: no cover
            pass
        listener.join(timeout=5.0)
        yf_logger.setLevel(previous_level)

    written = recorder.flush()
    n = recorder.n_messages
    logger.info("Recorded %d messages%s", n, f" to {written}" if written else "")
    return written, n


def option_contract_symbols(
    ticker: str,
    n_contracts: int,
    *,
    chain: pd.DataFrame | None = None,
    as_of: date | None = None,
) -> list[str]:
    """
    Most active option contracts of a ticker, for streaming alongside the spot.

    Parameters
    ----------
    ticker : str
        Underlying ticker.
    n_contracts : int
        How many contracts to return (ranked by open interest, then volume).
    chain : pd.DataFrame | None, optional
        An option chain to rank; the live chain is fetched when omitted.
    as_of : date | None, optional
        Only contracts expiring after this date are considered (default today).

    Returns
    -------
    list[str]
        Contract symbols such as ``SPY250919C00600000``.
    """
    if n_contracts <= 0:
        return []
    if chain is None:
        from .market_data_manager import get_live_option_chain

        chain = get_live_option_chain(ticker)
    if chain is None or chain.empty or "contractSymbol" not in chain.columns:
        return []
    as_of = as_of or date.today()
    live = chain[pd.to_datetime(chain["expiry"]).dt.date > as_of]
    ranked = live.sort_values(
        ["openInterest", "volume"], ascending=False, na_position="last"
    )
    return ranked["contractSymbol"].head(n_contracts).astype(str).tolist()
