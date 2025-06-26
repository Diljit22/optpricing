from __future__ import annotations

import json
import logging
import subprocess
from typing import Annotated

import typer

from quantfin.config import PROJECT_ROOT, _config
from quantfin.data import (
    get_available_snapshot_dates,
    load_market_snapshot,
    save_historical_returns,
    save_market_snapshot,
)
from quantfin.workflows import BacktestWorkflow, DailyWorkflow
from quantfin.workflows.configs import ALL_MODEL_CONFIGS

__doc__ = """
This module provides the command-line interface (CLI) for the quantfin library.

It uses the Typer library to create a user-friendly interface for running
calibrations, backtests, data management tasks, and launching the dashboard.
"""

# Create the main Typer application
app = typer.Typer(
    name="quantfin",
    help="A quantitative finance library for option pricing and analysis.",
    add_completion=False,
)

# Create a subcommand for data-related tasks
data_app = typer.Typer(name="data", help="Tools for downloading and managing data.")
app.add_typer(data_app)


# Utility function for logging setup
def setup_logging(verbose: bool):
    """
    Configures the root logger based on the verbosity flag.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


# CLI Commands


@app.command()
def dashboard():
    """
    Launches the Streamlit dashboard application.
    """
    # Note: We point directly to the source file now.
    app_path = PROJECT_ROOT / "src" / "quantfin" / "dashboard" / "app.py"
    typer.echo(f"Launching Streamlit dashboard from: {app_path}")

    if not app_path.exists():
        typer.secho(
            f"Error: Dashboard entry point not found at '{app_path}'.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)

    try:
        subprocess.run(
            ["streamlit", "run", str(app_path)],
            check=True,
        )
    except FileNotFoundError:
        typer.secho(
            "Error: 'streamlit' command not found; install: "
            "'pip install quantfin[app]'",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)


@data_app.command(name="download")
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
        tickers_to_download = _config.get("default_tickers", [])
        if not tickers_to_download:
            typer.secho(
                "Error: --all flag used, but no 'default_tickers' found "
                "in config.yaml.",
                fg=typer.colors.RED,
            )
            raise typer.Exit(code=1)
        typer.echo(f"Downloading all default tickers for period {period}...")
    elif tickers:
        tickers_to_download = tickers
        typer.echo(
            f"Downloading {period} historical data for tickers: "
            f"{', '.join(tickers_to_download)}"
        )
    else:
        typer.secho(
            "Error: Please provide at least one --ticker or use the --all flag.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)

    save_historical_returns(tickers_to_download, period=period)
    typer.secho("Download complete.", fg=typer.colors.GREEN)


@app.command()
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

    Saves the calibrated parameters to the 'artifacts/calibrated_params' directory.
    """
    setup_logging(verbose)

    # --- 1. Determine the snapshot date to use ---
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

    # --- 2. Load the market data ---
    market_data = load_market_snapshot(ticker, date)
    if market_data is None:
        typer.secho(
            f"Error: Failed to load market data for {ticker} on {date}.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)

    # --- 3. Parse fixed parameters ---
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

    # --- 4. Loop through and run workflow for each model ---
    for model_name in model:
        if model_name not in ALL_MODEL_CONFIGS:
            typer.secho(
                f"Warning: Model '{model_name}' not found. Skipping.",
                fg=typer.colors.YELLOW,
            )
            continue

        config = ALL_MODEL_CONFIGS[model_name].copy()
        config["ticker"] = ticker
        config["frozen"] = {**config.get("frozen", {}), **frozen_params}

        workflow = DailyWorkflow(market_data, config)
        workflow.run()

        # --- 5. Display and save results ---
        if workflow.results["Status"] == "Success":
            typer.secho(
                f"\nCalibration for {model_name} on {date} SUCCEEDED.",
                fg=typer.colors.GREEN,
            )
            typer.echo(f"  - Final RMSE: {workflow.results['RMSE']:.6f}")
            typer.echo(
                f"  - Calibrated Params: {workflow.results['Calibrated Params']}"
            )

            # Save the parameters
            params_to_save = {
                "model": model_name,
                "ticker": ticker,
                "date": date,
                "params": workflow.results["Calibrated Params"],
            }
            # Example filename: SPY_Heston_2023-01-01.json
            filename = f"{ticker}_{model_name}_{date}.json"
            save_path = PROJECT_ROOT / "artifacts" / "calibrated_params" / filename
            with open(save_path, "w") as f:
                json.dump(params_to_save, f, indent=4)
            typer.echo(f"  - Saved parameters to: {save_path}")

        else:
            typer.secho(
                f"\nCalibration for {model_name} on {date} FAILED.", fg=typer.colors.RED
            )
            typer.echo(f"  - Error: {workflow.results.get('Error', 'Unknown error')}")


@app.command()
def backtest(
    ticker: Annotated[
        str, typer.Option("--ticker", "-t", help="The stock ticker to backtest.")
    ],
    model: Annotated[
        str, typer.Option("--model", "-m", help="The single model to backtest.")
    ],
    # Other options like start/end date, fixed params can be added here
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Enable detailed logging.")
    ] = False,
):
    """
    Runs a historical backtest for a given model and ticker.
    """
    setup_logging(verbose)

    if model not in ALL_MODEL_CONFIGS:
        typer.secho(
            "Error: Model "
            f"'{model}' not found. Available models: "
            f"{list(ALL_MODEL_CONFIGS.keys())}",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)

    config = ALL_MODEL_CONFIGS[model].copy()
    config["ticker"] = ticker

    workflow = BacktestWorkflow(ticker, config)
    workflow.run()
    workflow.save_results()


@app.command()
def demo(
    model_name: Annotated[
        str | None,
        typer.Argument(help="Run a demo for a specific model name."),
    ] = None,
):
    """
    Runs a benchmark demo to showcase model and technique performance.

    If a model name is provided (e.g., 'BSM'), it runs only that benchmark.
    Otherwise, it runs the full suite.
    """
    benchmark_script_path = PROJECT_ROOT / "demo" / "benchmark.py"
    typer.echo(f"Executing benchmark demo from: {benchmark_script_path}")

    command_to_run = ["python", str(benchmark_script_path)]
    if model_name:
        command_to_run.append(model_name)

    try:
        subprocess.run(command_to_run, check=True)
    except FileNotFoundError:
        typer.secho(
            f"Error: Demo script not found at '{benchmark_script_path}'.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)


@data_app.command(name="snapshot")
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

    The data provider (yfinance or polygon) is determined by your config.yaml file.
    """
    if all_default:
        tickers_to_download = _config.get("default_tickers", [])
        if not tickers_to_download:
            typer.secho(
                "Error: --all flag used, but no 'default_tickers' in config.yaml.",
                fg=typer.colors.RED,
            )
            raise typer.Exit(code=1)
        typer.echo("Saving live market snapshots for all default tickers...")
    elif tickers:
        tickers_to_download = tickers
        typer.echo(
            f"Saving live market snapshots; tickers: {', '.join(tickers_to_download)}"
        )
    else:
        typer.secho(
            "Error: Please provide at least one --ticker or use the --all flag.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)

    save_market_snapshot(tickers_to_download)
    typer.secho("Snapshot complete.", fg=typer.colors.GREEN)
