from __future__ import annotations

from typing import Annotated, Any

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from optpricing.calibration import CalibrationSettings
from optpricing.workflows import BacktestWorkflow, load_tuned_settings

from .calibrate import select_models

__doc__ = """
CLI command for running historical backtests.
"""


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "-"
    return f"{value:.{digits}f}" if isinstance(value, int | float) else str(value)


def print_backtest_summary(summaries: list[dict[str, Any]], ticker: str) -> None:
    """Rich table of per-model backtest summaries (best out-of-sample first)."""
    console = Console()
    table = Table(title=f"Walk-forward backtest - {ticker}")
    for col, justify in (
        ("#", "right"),
        ("Model", "left"),
        ("Periods", "right"),
        ("OOS IV RMSE", "right"),
        ("In-sample IV RMSE", "right"),
        ("OOS price RMSE", "right"),
        ("BSM-ATM IV RMSE", "right"),
        ("Beats BSM-ATM", "right"),
        ("Time/period (s)", "right"),
    ):
        table.add_column(col, justify=justify)
    for i, s in enumerate(summaries, start=1):
        table.add_row(
            str(i),
            s["model"],
            f"{s['n_periods'] - s['n_failed']}/{s['n_periods']}",
            _fmt(s.get("mean_oos_iv_rmse")),
            _fmt(s.get("mean_in_sample_iv_rmse")),
            _fmt(s.get("mean_oos_price_rmse")),
            _fmt(s.get("mean_bsm_atm_iv_rmse")),
            _fmt(s.get("share_beating_bsm_atm"), 2),
            _fmt(s.get("mean_runtime_s"), 1),
        )
    console.print(table)


def backtest(
    ticker: Annotated[
        str, typer.Option("--ticker", "-t", help="The stock ticker to backtest.")
    ],
    model: Annotated[
        list[str] | None,
        typer.Option(
            "--model",
            "-m",
            help="Model to backtest ('all' for every calibratable model). "
            "Can be used multiple times.",
        ),
    ] = None,
    auto: Annotated[
        bool,
        typer.Option(
            "--auto", help="Use the settings tuned for this ticker (`optpricing tune`)."
        ),
    ] = False,
    allow_mc: Annotated[
        bool,
        typer.Option("--allow-mc", help="Also backtest simulation-only models (slow)."),
    ] = False,
    n_starts: Annotated[
        int, typer.Option("--n-starts", help="Starting points per calibration.")
    ] = 2,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Enable detailed logging.")
    ] = False,
):
    """
    Runs a walk-forward backtest (calibrate on day t, evaluate on day t+1).

    Results are saved per model to `artifacts/backtest_logs/` (per-period CSV,
    bucket CSV and a JSON summary) and summarised in a table.
    """
    from optpricing.cli.main import setup_logging

    setup_logging(verbose)
    ticker = ticker.upper()
    models = select_models(model or ["all"], allow_mc=allow_mc)
    base_settings = CalibrationSettings(n_starts=n_starts)

    summaries: list[dict[str, Any]] = []
    for model_name in models:
        from optpricing.workflows.configs import get_model_config

        config = get_model_config(model_name)
        settings = base_settings
        if auto:
            tuned = load_tuned_settings(ticker, model_name)
            if tuned is not None:
                settings = tuned
        typer.echo(f"Backtesting {model_name} ({settings.name} settings)...")
        workflow = BacktestWorkflow(
            ticker, config, settings=settings, allow_mc=allow_mc
        )
        workflow.run()
        saved = workflow.save_results()
        summary = workflow.summary()
        summaries.append(summary)
        if saved:
            typer.echo(f"  - Saved: {saved[0]}")
        elif summary["n_periods"] == 0:
            typer.secho(
                "  - Not enough snapshots for a backtest (need at least two dates).",
                fg=typer.colors.YELLOW,
            )

    summaries.sort(
        key=lambda s: (
            s.get("mean_oos_iv_rmse")
            if s.get("mean_oos_iv_rmse") is not None
            else float("inf")
        )
    )
    print_backtest_summary(summaries, ticker)
