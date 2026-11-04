from __future__ import annotations

import json
import logging
from typing import Annotated, Any

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from optpricing.calibration import CalibrationSettings
from optpricing.config import CALIBRATED_PARAMS_DIR, CALIBRATION_LOGS_DIR
from optpricing.data import get_available_snapshot_dates, load_market_snapshot
from optpricing.workflows import DailyWorkflow, load_tuned_settings
from optpricing.workflows.configs import (
    calibratable_model_names,
    get_model_config,
    resolve_model_name,
)

__doc__ = """
CLI command for model calibration.
"""

logger = logging.getLogger(__name__)


def parse_fixed_params(fix_param: list[str] | None) -> dict[str, float]:
    """Parses ``key=value`` strings into a dict, exiting with a message on error."""
    frozen: dict[str, float] = {}
    for p in fix_param or []:
        try:
            key, value = p.split("=", 1)
            frozen[key.strip()] = float(value)
        except ValueError:
            typer.secho(
                f"Invalid format for fixed parameter: '{p}'. Use 'key=value'.",
                fg=typer.colors.RED,
            )
            raise typer.Exit(code=1)
    return frozen


def select_models(model: list[str], *, allow_mc: bool) -> list[str]:
    """Expands ``all`` and aliases; exits on unknown names."""
    if any(m.lower() == "all" for m in model):
        return calibratable_model_names(allow_mc=allow_mc)
    names: list[str] = []
    for m in model:
        try:
            names.append(resolve_model_name(m))
        except KeyError as e:
            typer.secho(str(e.args[0]), fg=typer.colors.RED)
            raise typer.Exit(code=1)
    return names


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "-"
    if isinstance(value, int | float):
        return f"{value:.{digits}f}"
    return str(value)


def print_leaderboard(rows: list[dict[str, Any]], title: str) -> None:
    """Rich table of per-model calibration results."""
    console = Console()
    table = Table(title=title)
    for col, justify in (
        ("#", "right"),
        ("Model", "left"),
        ("Status", "left"),
        ("Calib IV RMSE", "right"),
        ("Eval IV RMSE", "right"),
        ("Price RMSE", "right"),
        ("Options", "right"),
        ("Time (s)", "right"),
        ("Settings", "left"),
    ):
        table.add_column(col, justify=justify)
    for i, r in enumerate(rows, start=1):
        table.add_row(
            str(i),
            str(r["Model"]),
            "OK" if r.get("Status") == "Success" else "FAIL",
            _fmt(r.get("Calibration IV RMSE")),
            _fmt(r.get("IV RMSE")),
            _fmt(r.get("RMSE")),
            f"{r.get('N Calibration Options', 0) or 0}/{r.get('N Options', 0) or 0}",
            _fmt(r.get("Runtime (s)"), 1),
            str(r.get("Settings", "")),
        )
    console.print(table)


def calibrate(
    ticker: Annotated[
        str,
        typer.Option("--ticker", "-t", help="The stock ticker to calibrate against."),
    ],
    model: Annotated[
        list[str] | None,
        typer.Option(
            "--model",
            "-m",
            help="Model to calibrate ('all' for every calibratable model). "
            "Can be used multiple times.",
        ),
    ] = None,
    date: Annotated[
        str | None,
        typer.Option(
            "--date",
            "-d",
            help="Snapshot date (YYYY-MM-DD). Defaults to latest available.",
        ),
    ] = None,
    fix_param: Annotated[
        list[str] | None,
        typer.Option(
            "--fix",
            help="Fix a parameter (e.g., 'sigma=0.25'). Can be used multiple times.",
        ),
    ] = None,
    auto: Annotated[
        bool,
        typer.Option(
            "--auto",
            help="Use the settings tuned for this ticker (`optpricing tune`).",
        ),
    ] = False,
    allow_mc: Annotated[
        bool,
        typer.Option(
            "--allow-mc", help="Also calibrate simulation-only models (slow)."
        ),
    ] = False,
    objective: Annotated[
        str,
        typer.Option("--objective", help="Residual space: 'iv' or 'price'."),
    ] = "iv",
    n_starts: Annotated[
        int, typer.Option("--n-starts", help="Starting points per model.")
    ] = 4,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Enable detailed logging.")
    ] = False,
):
    """
    Calibrates one or more models to a market snapshot and ranks them.

    With `--model all` every model that can be priced on a chain is fitted and a
    leaderboard (sorted by evaluation IV RMSE) is printed and saved to
    `artifacts/calibration_logs/`; parameters go to `artifacts/calibrated_params/`.
    """
    from optpricing.cli.main import setup_logging

    setup_logging(verbose)
    ticker = ticker.upper()
    models = select_models(model or ["all"], allow_mc=allow_mc)

    if date is None:
        typer.echo(f"No date specified for {ticker}. Finding latest snapshot...")
        available_dates = get_available_snapshot_dates(ticker)
        if not available_dates:
            typer.secho(
                f"Error: No market data snapshots found for ticker '{ticker}'.",
                fg=typer.colors.RED,
            )
            raise typer.Exit(code=1)
        date = available_dates[0]
        typer.echo(f"Using latest date: {date}")

    market_data = load_market_snapshot(ticker, date)
    if market_data is None:
        typer.secho(
            f"Error: Failed to load market data for {ticker} on {date}.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)

    frozen_params = parse_fixed_params(fix_param)
    try:
        base_settings = CalibrationSettings(objective=objective, n_starts=n_starts)
    except ValueError as e:
        typer.secho(f"Error: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    CALIBRATED_PARAMS_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for model_name in models:
        config = get_model_config(model_name)
        config["ticker"] = ticker
        config["frozen"] = {**config.get("frozen", {}), **frozen_params}
        settings = base_settings
        if auto:
            tuned = load_tuned_settings(ticker, model_name)
            if tuned is not None:
                settings = (
                    tuned.replace(objective=objective)
                    if objective != tuned.objective
                    else tuned
                )
            else:
                typer.secho(
                    f"No tuned settings for {ticker}/{model_name}; using defaults.",
                    fg=typer.colors.YELLOW,
                )

        typer.echo(f"Calibrating {model_name} on {date} ({settings.name} settings)...")
        workflow = DailyWorkflow(
            market_data, config, settings=settings, allow_mc=allow_mc
        )
        workflow.run()
        result = dict(workflow.results)
        result["Settings"] = settings.name
        result.pop("Buckets", None)
        rows.append(result)

        if result["Status"] == "Success":
            params_to_save = {
                "model": model_name,
                "ticker": ticker,
                "date": date,
                "params": {k: float(v) for k, v in result["Calibrated Params"].items()},
                "rate": result["Implied Rate"],
                "dividend_yield": result["Known Dividend"],
                "calibration_iv_rmse": result["Calibration IV RMSE"],
                "evaluation_iv_rmse": result["IV RMSE"],
                "evaluation_price_rmse": result["RMSE"],
                "settings": settings.to_dict(),
            }
            save_path = CALIBRATED_PARAMS_DIR / f"{ticker}_{model_name}_{date}.json"
            with open(save_path, "w") as f:
                json.dump(params_to_save, f, indent=4)
            typer.echo(f"  - Saved parameters to: {save_path}")
        else:
            typer.secho(
                f"  - {model_name} FAILED: {result.get('Error', 'Unknown error')}",
                fg=typer.colors.RED,
            )

    rows.sort(
        key=lambda r: (
            r["Status"] != "Success",
            r.get("IV RMSE")
            if pd.notna(r.get("IV RMSE", float("nan")))
            else float("inf"),
        )
    )
    print_leaderboard(rows, f"Calibration leaderboard - {ticker} {date}")
    if any(r["Status"] == "Success" for r in rows):
        CALIBRATION_LOGS_DIR.mkdir(parents=True, exist_ok=True)
        out = CALIBRATION_LOGS_DIR / f"{ticker}_leaderboard_{date}.csv"
        frame = pd.DataFrame(rows)
        if "Calibrated Params" in frame.columns:
            frame["Calibrated Params"] = frame["Calibrated Params"].map(
                lambda p: json.dumps(p) if isinstance(p, dict) else p
            )
        frame.to_csv(out, index=False)
        typer.echo(f"Leaderboard saved to: {out}")
    if not any(r["Status"] == "Success" for r in rows):
        raise typer.Exit(code=1)
