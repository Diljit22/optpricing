from __future__ import annotations

import logging

import typer

from .commands.backtest import backtest
from .commands.calibrate import calibrate
from .commands.dashboard import dashboard
from .commands.data import download_data, get_dividends, save_snapshot
from .commands.demo import demo as demo_app
from .commands.price import price
from .commands.tools import get_implied_rate

__doc__ = """
This module provides the main entry point for the optpricing CLI.

It defines the main Typer application and registers all commands and subcommands
from the `commands` directory.
"""

app = typer.Typer(
    name="optpricing",
    help="A quantitative finance library for option pricing and analysis.",
    add_completion=False,
)

data_app = typer.Typer(name="data", help="Tools for downloading and managing data.")
tools_app = typer.Typer(name="tools", help="Miscellaneous financial utility tools.")


app.command()(dashboard)
app.command()(calibrate)
app.command()(backtest)
app.command()(price)
app.add_typer(demo_app)


data_app.command(name="download")(download_data)
data_app.command(name="snapshot")(save_snapshot)
data_app.command(name="dividends")(get_dividends)

tools_app.command(name="implied-rate")(get_implied_rate)


app.add_typer(data_app)
app.add_typer(tools_app)


def setup_logging(verbose: bool):
    """
    Configures the root logger based on the verbosity flag.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
