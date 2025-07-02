from __future__ import annotations

from typing import Annotated

import typer

__doc__ = """
CLI command for running benchmark demos.
"""


def demo(
    model_name: Annotated[
        str | None,
        typer.Argument(help="Run a demo for a specific model name."),
    ] = None,
):
    """
    Runs a benchmark demo to showcase model and technique performance.
    """
    typer.secho(
        "The 'demo' command for developers and requires the full source repository.",
        fg=typer.colors.YELLOW,
    )
    typer.echo("Please run 'make demo' from the project's root directory instead.")
    raise typer.Exit()
