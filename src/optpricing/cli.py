from __future__ import annotations

import json
import logging
import subprocess
from importlib import resources
from pathlib import Path
from typing import Annotated, Optional

import numpy as np
import pandas as pd
import typer

from optpricing.atoms import Option, OptionType, Rate, Stock
from optpricing.calibration import fit_rate_and_dividend
from optpricing.calibration.technique_selector import select_fastest_technique
from optpricing.config import _config
from optpricing.data import (
    get_available_snapshot_dates,
    get_live_option_chain,
    load_market_snapshot,
    save_historical_returns,
    save_market_snapshot,
)
from optpricing.models import BaseModel
from optpricing.parity import ImpliedRateModel
from optpricing.workflows import BacktestWorkflow, DailyWorkflow
from optpricing.workflows.configs import ALL_MODEL_CONFIGS

__doc__ = """
This module provides the command-line interface (CLI) for the optpricing library.

It uses the Typer library to create a user-friendly interface for running
calibrations, backtests, data management tasks, and launching the dashboard.
"""

# Create the main Typer application
app = typer.Typer(
    name="optpricing",
    help="A quantitative finance library for option pricing and analysis.",
    add_completion=False,
)

# Create a subcommand for data-related tasks
data_app = typer.Typer(name="data", help="Tools for downloading and managing data.")
app.add_typer(data_app)

tools_app = typer.Typer(name="tools", help="Miscellaneous financial utility tools.")
app.add_typer(tools_app)


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
    try:
        with resources.path("optpricing.dashboard", "app.py") as app_path:
            typer.echo(f"Launching Streamlit dashboard from: {app_path}")
            subprocess.run(["streamlit", "run", str(app_path)], check=True)
    except FileNotFoundError:
        typer.secho(
            "Error: 'streamlit' command not found; 'pip install optpricing[app]'",
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

    current_dir = Path.cwd()
    artifacts_base_dir = current_dir / _config.get("artifacts_directory", "artifacts")
    calibrated_params_dir = artifacts_base_dir / "calibrated_params"
    calibrated_params_dir.mkdir(parents=True, exist_ok=True)

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
            save_path = calibrated_params_dir / filename
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
    typer.secho(
        "The 'demo' command for developers and requires the full source repository.",
        fg=typer.colors.YELLOW,
    )
    typer.echo("Please run 'make demo' from the project's root directory instead.")
    raise typer.Exit()


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

    The data provider (yfinance) is determined by the config.yaml file.
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


@app.command()
def price(
    ticker: Annotated[
        str, typer.Option("--ticker", "-t", help="Stock ticker for the option.")
    ],
    strike: Annotated[
        float, typer.Option("--strike", "-k", help="Strike price of the option.")
    ],
    maturity: Annotated[
        str,
        typer.Option("--maturity", "-T", help="Maturity date in YYYY-MM-DD format."),
    ],
    option_type: Annotated[
        str, typer.Option("--type", help="Option type: 'call' or 'put'.")
    ] = "call",
    model: Annotated[
        str, typer.Option("--model", "-m", help="The model to use for pricing.")
    ] = "BSM",
    param: Annotated[
        list[str] | None,
        typer.Option(
            "--param",
            help="Set a model parameter (e.g., 'sigma=0.2'). Can use multiple times.",
        ),
    ] = None,
):
    """Prices a single option using live market data and user model parameters."""
    msg = (
        f"Pricing {ticker} {strike} {option_type.upper()} expiring {maturity} "
        f"using {model} model..."
    )
    typer.echo(msg)

    # 1. Parse model parameters
    model_params = {}
    if param:
        for p in param:
            try:
                key, value = p.split("=")
                model_params[key.strip()] = float(value)
            except ValueError:
                typer.secho(
                    f"Invalid format for parameter: '{p}'. Use 'key=value'.",
                    fg=typer.colors.RED,
                )
                raise typer.Exit(code=1)

    # 2. Get live market data for r, q, and spot
    typer.echo("Fetching live market data...")
    live_chain = get_live_option_chain(ticker)
    if live_chain is None or live_chain.empty:
        typer.secho(
            f"Error: Could not fetch live option chain for {ticker}.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)

    spot = live_chain["spot_price"].iloc[0]
    calls = live_chain[live_chain["optionType"] == "call"]
    puts = live_chain[live_chain["optionType"] == "put"]
    r, q = fit_rate_and_dividend(calls, puts, spot)
    typer.echo(
        f"Live Data -> Spot: {spot:.2f}, Implied Rate: {r:.4f}, Implied Div: {q:.4f}"
    )

    # 3. Create objects and price
    stock = Stock(spot=spot, dividend=q)
    rate = Rate(rate=r)
    maturity_years = (pd.to_datetime(maturity) - pd.Timestamp.now()).days / 365.25
    option = Option(
        strike=strike,
        maturity=maturity_years,
        option_type=OptionType[option_type.upper()],
    )

    model_class = ALL_MODEL_CONFIGS[model]["model_class"]
    model_instance = model_class(params=model_params)
    technique = select_fastest_technique(model_instance)

    # Prepare kwargs for techniques that need extra info (e.g., Heston's v0)
    pricing_kwargs = model_params.copy()

    price_result = technique.price(
        option, stock, model_instance, rate, **pricing_kwargs
    )
    delta = technique.delta(option, stock, model_instance, rate, **pricing_kwargs)
    gamma = technique.gamma(option, stock, model_instance, rate, **pricing_kwargs)
    vega = technique.vega(option, stock, model_instance, rate, **pricing_kwargs)

    # 4. Display results
    typer.secho("\n--- Pricing Results ---", fg=typer.colors.CYAN)
    typer.echo(f"Price: {price_result.price:.4f}")
    typer.echo(f"Delta: {delta:.4f}")
    typer.echo(f"Gamma: {gamma:.4f}")
    typer.echo(f"Vega:  {vega:.4f}")


@tools_app.command(name="implied-rate")
def get_implied_rate(
    ticker: Annotated[
        str, typer.Option("--ticker", "-t", help="Stock ticker for the option pair.")
    ],
    strike: Annotated[
        float, typer.Option("--strike", "-k", help="Strike price of the option pair.")
    ],
    maturity: Annotated[
        str,
        typer.Option("--maturity", "-T", help="Maturity date in YYYY-MM-DD format."),
    ],
):
    """Calculates the implied risk-free rate from a live call-put pair."""
    typer.echo(
        f"Fetching live prices for {ticker} {strike} options expiring {maturity}..."
    )

    live_chain = get_live_option_chain(ticker)
    if live_chain is None or live_chain.empty:
        typer.secho(
            f"Error: Could not fetch live option chain for {ticker}.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)

    maturity_dt = pd.to_datetime(maturity).date()
    chain_for_expiry = live_chain[live_chain["expiry"].dt.date == maturity_dt]

    call_option = chain_for_expiry[
        (chain_for_expiry["strike"] == strike)
        & (chain_for_expiry["optionType"] == "call")
    ]
    put_option = chain_for_expiry[
        (chain_for_expiry["strike"] == strike)
        & (chain_for_expiry["optionType"] == "put")
    ]

    if call_option.empty or put_option.empty:
        typer.secho(
            f"Error: Did not find both: call & put for strike {strike} on {maturity}.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)

    call_price = call_option["marketPrice"].iloc[0]
    put_price = put_option["marketPrice"].iloc[0]

    spot_price = call_option["spot_price"].iloc[0]
    maturity_years = call_option["maturity"].iloc[0]

    pair_msg = (
        f"Found Pair -> Call Price: {call_price:.2f}, Put Price: {put_price:.2f}, "
        f"Spot: {spot_price:.2f}"
    )
    typer.echo(pair_msg)

    implied_rate_model = ImpliedRateModel(params={})
    try:
        implied_r = implied_rate_model.price_closed_form(
            call_price=call_price,
            put_price=put_price,
            spot=spot_price,
            strike=strike,
            t=maturity_years,
            q=0,
        )
        typer.secho(
            f"\nImplied Risk-Free Rate (r): {implied_r:.4%}", fg=typer.colors.GREEN
        )
    except Exception as e:
        typer.secho(f"\nError calculating implied rate: {e}", fg=typer.colors.RED)
