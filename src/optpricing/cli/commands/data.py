from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from optpricing.config import _config
from optpricing.data import (
    get_live_dividend_yield,
    save_historical_returns,
    save_market_snapshot,
)

__doc__ = """
CLI commands for downloading and managing market data.
"""


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
):
    """
    Downloads and saves historical log returns for specified tickers or all defaults.
    """
    if all_default:
        tickers_to_process = _config.get("default_tickers", [])
        if not tickers_to_process:
            typer.secho(
                "Error: --all flag used, but no 'default_tickers' found "
                "in config.yaml.",
                fg=typer.colors.RED,
            )
            raise typer.Exit(code=1)
        typer.echo(f"Downloading all default tickers for period {period}...")
    elif tickers:
        tickers_to_process = tickers
        typer.echo(
            f"Downloading {period} historical data for tickers: "
            f"{', '.join(tickers_to_process)}"
        )
    else:
        typer.secho(
            "Error: Please provide at least one --ticker or use the --all flag.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)

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
):
    """
    Fetches and saves a live market data snapshot for specified tickers.
    """
    if all_default:
        tickers_to_process = _config.get("default_tickers", [])
        if not tickers_to_process:
            typer.secho(
                "Error: --all flag used, but no 'default_tickers' in config.yaml.",
                fg=typer.colors.RED,
            )
            raise typer.Exit(code=1)
        typer.echo("Saving live market snapshots for all default tickers...")
    elif tickers:
        tickers_to_process = tickers
        typer.echo(
            f"Saving live market snapshots; tickers: {', '.join(tickers_to_process)}"
        )
    else:
        typer.secho(
            "Error: Please provide at least one --ticker or use the --all flag.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)

    save_market_snapshot(tickers_to_process)
    typer.secho("Snapshot complete.", fg=typer.colors.GREEN)


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
    if all_default:
        tickers_to_fetch = _config.get("default_tickers", [])
        if not tickers_to_fetch:
            typer.secho(
                "Error: --all flag used, but no 'default_tickers' in config.yaml.",
                fg=typer.colors.RED,
            )
            raise typer.Exit(code=1)
        typer.echo("Fetching dividend yields for all default tickers...")
    elif tickers:
        tickers_to_fetch = tickers
    else:
        typer.secho(
            "Error: Please provide at least one --ticker or use the --all flag.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)

    console = Console()
    table = Table(title="Live Dividend Yields")
    table.add_column("Ticker", justify="left", style="cyan", no_wrap=True)
    table.add_column("Dividend Yield", justify="right", style="magenta")

    for ticker in tickers_to_fetch:
        yield_val = get_live_dividend_yield(ticker)
        table.add_row(ticker.upper(), f"{yield_val:.4%}")

    console.print(table)
