from __future__ import annotations

from typing import Annotated

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from optpricing.workflows import CANDIDATE_SETTINGS, TuningResult, tune_ticker

from .calibrate import select_models

__doc__ = """
CLI command for per-ticker auto-tuning of the calibration settings.
"""


def print_tuning_leaderboard(result: TuningResult) -> None:
    """Rich table of the tuning leaderboard (best model first)."""
    console = Console()
    table = Table(
        title=f"Tuning leaderboard - {result.ticker} "
        f"({len(result.dates)} snapshots, {result.dates[0]} to {result.dates[-1]})"
    )
    for col, justify in (
        ("#", "right"),
        ("Model", "left"),
        ("Best settings", "left"),
        ("OOS IV RMSE", "right"),
        ("In-sample IV RMSE", "right"),
        ("OOS price RMSE", "right"),
        ("BSM-ATM IV RMSE", "right"),
        ("Periods", "right"),
        ("Time (s)", "right"),
    ):
        table.add_column(col, justify=justify)
    for row in result.leaderboard:
        table.add_row(
            str(row.get("rank", "")),
            row["model"],
            row["settings"],
            f"{row['oos_iv_rmse']:.4f}",
            f"{row['in_sample_iv_rmse']:.4f}"
            if pd.notna(row.get("in_sample_iv_rmse"))
            else "-",
            f"{row['oos_price_rmse']:.3f}"
            if pd.notna(row.get("oos_price_rmse"))
            else "-",
            f"{row['bsm_atm_iv_rmse']:.4f}"
            if pd.notna(row.get("bsm_atm_iv_rmse"))
            else "-",
            f"{row['n_periods'] - row['n_failed']}/{row['n_periods']}",
            f"{row['runtime_s']:.1f}",
        )
    console.print(table)


def tune(
    ticker: Annotated[
        str, typer.Option("--ticker", "-t", help="The stock ticker to tune for.")
    ],
    model: Annotated[
        list[str] | None,
        typer.Option(
            "--model",
            "-m",
            help="Model to include ('all' by default). Can be used multiple times.",
        ),
    ] = None,
    candidates: Annotated[
        str,
        typer.Option(
            "--candidates",
            help="'quick' (3 settings), 'full' (all) or a comma-separated list of: "
            + ", ".join(CANDIDATE_SETTINGS),
        ),
    ] = "quick",
    max_pairs: Annotated[
        int | None,
        typer.Option("--max-pairs", help="Use only the most recent N day pairs."),
    ] = None,
    allow_mc: Annotated[
        bool,
        typer.Option("--allow-mc", help="Include simulation-only models (slow)."),
    ] = False,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Enable detailed logging.")
    ] = False,
):
    """
    Tunes calibration settings per model by walk-forward cross-validation and
    ranks the models for a ticker.

    The result is cached in `artifacts/tuning/{TICKER}.json` and used by
    `optpricing calibrate --auto` and `optpricing backtest --auto`.
    """
    from optpricing.cli.main import setup_logging

    setup_logging(verbose)
    models = select_models(model or ["all"], allow_mc=allow_mc)
    candidate_arg: str | list[str]
    if candidates in ("quick", "full"):
        candidate_arg = candidates
    else:
        candidate_arg = [c.strip() for c in candidates.split(",") if c.strip()]

    try:
        result = tune_ticker(
            ticker,
            models=models,
            candidates=candidate_arg,
            max_pairs=max_pairs,
            allow_mc=allow_mc,
            progress=lambda msg: typer.echo(f"  tuning {msg}..."),
        )
    except ValueError as e:
        typer.secho(f"Error: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    if not result.leaderboard:
        typer.secho("No model could be tuned.", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    print_tuning_leaderboard(result)
    typer.secho(
        f"Best model for {result.ticker}: {result.best_model}. "
        f"Settings cached in {result.save()}",
        fg=typer.colors.GREEN,
    )
