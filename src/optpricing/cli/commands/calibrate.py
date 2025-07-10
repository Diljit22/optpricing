from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from optpricing.config import _config
from optpricing.data import get_available_snapshot_dates, load_market_snapshot
from optpricing.workflows import DailyWorkflow
from optpricing.workflows.configs import ALL_MODEL_CONFIGS

__doc__ = """
CLI command for model calibration.
"""

AVAILABLE_MODELS_FOR_CALIBRATION = {
    name: config
    for name, config in ALL_MODEL_CONFIGS.items()
    if name in ["BSM", "Merton"]
}


def calibrate(
    ticker: Annotated[
        str,
        typer.Option("--ticker", "-t", help="The stock ticker to calibrate against."),
    ],
    model: Annotated[
        list[str],
        typer.Option(
            "--model", "-m", help="Model to calibrate. Can be used multiple times."
        ),
    ],
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
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Enable detailed logging.")
    ] = False,
):
    """
    Calibrates one or more models to market data for a given ticker and date.
    """
    from optpricing.cli.main import setup_logging

    setup_logging(verbose)

    current_dir = Path.cwd()
    artifacts_base_dir = current_dir / _config.get("artifacts_directory", "artifacts")
    calibrated_params_dir = artifacts_base_dir / "calibrated_params"
    calibrated_params_dir.mkdir(parents=True, exist_ok=True)

    if date is None:
        typer.echo(
            f"No date specified for {ticker}. Finding latest available snapshot..."
        )
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

    frozen_params = {}
    if fix_param:
        for p in fix_param:
            try:
                key, value = p.split("=")
                frozen_params[key.strip()] = float(value)
            except ValueError:
                typer.secho(
                    f"Invalid format for fixed parameter: '{p}'. Use 'key=value'.",
                    fg=typer.colors.RED,
                )
                raise typer.Exit(code=1)

    for model_name in model:
        if model_name not in AVAILABLE_MODELS_FOR_CALIBRATION:
            typer.secho(
                f"Warning: Model '{model_name}' is not supported for calibration. "
                f"Available: {list(AVAILABLE_MODELS_FOR_CALIBRATION.keys())}. "
                f"Skipping.",
                fg=typer.colors.YELLOW,
            )
            continue

        config = AVAILABLE_MODELS_FOR_CALIBRATION[model_name].copy()
        config["ticker"] = ticker
        config["frozen"] = {**config.get("frozen", {}), **frozen_params}

        workflow = DailyWorkflow(market_data, config)
        workflow.run()

        if workflow.results["Status"] == "Success":
            typer.secho(
                f"\nCalibration for {model_name} on {date} SUCCEEDED.",
                fg=typer.colors.GREEN,
            )
            typer.echo(f"  - Final RMSE: {workflow.results['RMSE']:.6f}")
            typer.echo(
                f"  - Calibrated Params: {workflow.results['Calibrated Params']}"
            )

            params_to_save = {
                "model": model_name,
                "ticker": ticker,
                "date": date,
                "params": workflow.results["Calibrated Params"],
            }
            filename = f"{ticker}_{model_name}_{date}.json"
            save_path = calibrated_params_dir / filename
            with open(save_path, "w") as f:
                json.dump(params_to_save, f, indent=4)
            typer.echo(f"  - Saved parameters to: {save_path}")
        else:
            typer.secho(
                f"\nCalibration for {model_name} on {date} FAILED.",
                fg=typer.colors.RED,
            )
            typer.echo(f"  - Error: {workflow.results.get('Error', 'Unknown error')}")
