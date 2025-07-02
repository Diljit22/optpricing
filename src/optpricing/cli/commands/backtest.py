from __future__ import annotations

from typing import Annotated

import typer

from optpricing.workflows import BacktestWorkflow
from optpricing.workflows.configs import ALL_MODEL_CONFIGS

__doc__ = """
CLI command for running historical backtests.
"""


def backtest(
    ticker: Annotated[
        str, typer.Option("--ticker", "-t", help="The stock ticker to backtest.")
    ],
    model: Annotated[
        str, typer.Option("--model", "-m", help="The single model to backtest.")
    ],
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Enable detailed logging.")
    ] = False,
):
    """
    Runs a historical backtest for a given model and ticker.
    """
    from optpricing.cli.main import setup_logging

    setup_logging(verbose)

    if model not in ALL_MODEL_CONFIGS:
        typer.secho(
            f"Error: Model '{model}' not found. "
            f"Available: {list(ALL_MODEL_CONFIGS.keys())}",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)

    config = ALL_MODEL_CONFIGS[model].copy()
    config["ticker"] = ticker

    workflow = BacktestWorkflow(ticker, config)
    workflow.run()
    workflow.save_results()
