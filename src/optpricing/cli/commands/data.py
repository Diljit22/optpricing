from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from optpricing.config import _config
from optpricing.data import (
    get_live_dividend_yield,
    option_contract_symbols,
    record_live_quotes,
    save_historical_returns,
    save_market_snapshot,
    save_risk_free_history,
)

__doc__ = """
CLI commands for downloading and managing market data.
"""


def _resolve_tickers(
    tickers: list[str] | None, all_default: bool, what: str
) -> list[str]:
    if all_default:
        selected = _config.get("default_tickers", [])
        if not selected:
            typer.secho(
                "Error: --all flag used, but no 'default_tickers' in config.yaml.",
                fg=typer.colors.RED,
            )
            raise typer.Exit(code=1)
        typer.echo(f"{what} for all default tickers...")
        return list(selected)
    if tickers:
        typer.echo(f"{what} for tickers: {', '.join(tickers)}")
        return list(tickers)
    typer.secho(
        "Error: Please provide at least one --ticker or use the --all flag.",
        fg=typer.colors.RED,
    )
    raise typer.Exit(code=1)


def download_data(
    tickers: Annotated[
        list[str] | None,
        typer.Option(
            "--ticker",
            "-t",
            help="Stock ticker to download. Can be used multiple times.",
        ),
    ] = None,
    all_default: Annotated[
        bool,
        typer.Option(
            "--all", help="Download all default tickers specified in config.yaml."
        ),
    ] = False,
    period: Annotated[
        str,
        typer.Option(
            "--period",
            "-p",
            help="Time period for historical data (e.g., '10y', '5y').",
        ),
    ] = "10y",
    rates: Annotated[
        bool,
        typer.Option(
            "--rates",
            help="Also download the ^IRX (13-week T-bill) history used as the "
            "risk-free rate of older snapshots.",
        ),
    ] = False,
):
    """
    Downloads historical prices, dividends and log returns (and optionally the
    risk-free rate history) for specified tickers or all defaults.
    """
    if rates:
        typer.echo(f"Downloading {period} risk-free (^IRX) history...")
        save_risk_free_history(period=period)
        if not tickers and not all_default:
            typer.secho("Download complete.", fg=typer.colors.GREEN)
            return
    tickers_to_process = _resolve_tickers(
        tickers, all_default, f"Downloading {period} historical data"
    )
    save_historical_returns(tickers_to_process, period=period)
    typer.secho("Download complete.", fg=typer.colors.GREEN)


def save_snapshot(
    tickers: Annotated[
        list[str] | None,
        typer.Option(
            "--ticker",
            "-t",
            help="Stock ticker to snapshot. Can be used multiple times.",
        ),
    ] = None,
    all_default: Annotated[
        bool,
        typer.Option(
            "--all", help="Snapshot all default tickers specified in config.yaml."
        ),
    ] = False,
    compact: Annotated[
        bool,
        typer.Option(
            "--compact",
            help="Keep only liquid options and compress (what the scheduled "
            "GitHub Action commits; ~10x smaller files).",
        ),
    ] = False,
    only_fresh: Annotated[
        bool,
        typer.Option(
            "--only-fresh",
            help="Skip tickers whose chain shows no trade today (holidays).",
        ),
    ] = False,
):
    """
    Fetches and saves a live market data snapshot (schema v2: spot, rate and
    dividend yield included) for specified tickers.
    """
    tickers_to_process = _resolve_tickers(
        tickers, all_default, "Saving live market snapshots"
    )
    written = save_market_snapshot(
        tickers_to_process, compact=compact, only_fresh=only_fresh
    )
    for path in written:
        typer.echo(f"  - {path}")
    if not written:
        typer.secho("No snapshot was written.", fg=typer.colors.YELLOW)
        raise typer.Exit(code=1)
    typer.secho(f"Snapshot complete ({len(written)} file(s)).", fg=typer.colors.GREEN)


def stream_quotes(
    tickers: Annotated[
        list[str] | None,
        typer.Option(
            "--ticker",
            "-t",
            help="Underlying to stream (repeatable).",
        ),
    ] = None,
    symbols: Annotated[
        list[str] | None,
        typer.Option(
            "--symbol",
            "-s",
            help="Any extra Yahoo symbol, e.g. an option contract "
            "'SPY250919C00600000' or an index '^SPX' (repeatable).",
        ),
    ] = None,
    duration: Annotated[
        float,
        typer.Option("--duration", "-d", help="Recording length in seconds."),
    ] = 60.0,
    contracts: Annotated[
        int,
        typer.Option(
            "--contracts",
            "-c",
            help="Also stream the N most active option contracts of each ticker.",
        ),
    ] = 0,
    output: Annotated[
        Path | None,
        typer.Option(
            "--output", "-o", help="Parquet file (default: data/live_quotes/)."
        ),
    ] = None,
):
    """
    Records live quotes from Yahoo Finance's free streamer into a parquet file.

    Press Ctrl-C to stop early; whatever was received is saved.
    """
    requested = [t.upper() for t in (tickers or [])] + [
        s.upper() for s in (symbols or [])
    ]
    if not requested:
        typer.secho(
            "Error: provide at least one --ticker or --symbol.", fg=typer.colors.RED
        )
        raise typer.Exit(code=1)
    if contracts > 0:
        for t in tickers or []:
            found = option_contract_symbols(t, contracts)
            typer.echo(f"  {t}: streaming {len(found)} option contract(s)")
            requested.extend(found)
    seen: set[str] = set()
    all_symbols = [s for s in requested if not (s in seen or seen.add(s))]
    typer.echo(f"Streaming {len(all_symbols)} symbol(s) for {duration:.0f}s...")
    path, n = record_live_quotes(all_symbols, duration, output)
    if path is None:
        typer.secho("No quotes received (market closed?).", fg=typer.colors.YELLOW)
        raise typer.Exit(code=1)
    typer.secho(f"Recorded {n} quotes to {path}", fg=typer.colors.GREEN)


def get_dividends(
    tickers: Annotated[
        list[str] | None,
        typer.Option(
            "--ticker",
            "-t",
            help="Stock ticker to fetch. Can be used multiple times.",
        ),
    ] = None,
    all_default: Annotated[
        bool,
        typer.Option(
            "--all", help="Fetch for all default tickers specified in config.yaml."
        ),
    ] = False,
):
    """
    Fetches and displays the live forward dividend yield for specified tickers.
    """
    tickers_to_fetch = _resolve_tickers(
        tickers, all_default, "Fetching dividend yields"
    )

    console = Console()
    table = Table(title="Live Dividend Yields")
    table.add_column("Ticker", justify="left", style="cyan", no_wrap=True)
    table.add_column("Dividend Yield", justify="right", style="magenta")

    for ticker in tickers_to_fetch:
        yield_val = get_live_dividend_yield(ticker)
        table.add_row(ticker.upper(), f"{yield_val:.4%}")

    console.print(table)
