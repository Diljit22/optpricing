from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Annotated

import typer

__doc__ = """
CLI command for running benchmark demos.
"""


def demo(
    model: Annotated[
        str | None,
        typer.Option(
            "--model",
            "-m",
            help="Run benchmark for a specific model name (e.g., 'BSM').",
            case_sensitive=False,
        ),
    ] = None,
    technique: Annotated[
        str | None,
        typer.Option(
            "--technique",
            "-t",
            help="Run benchmark for a specific technique (e.g., 'MC').",
            case_sensitive=False,
        ),
    ] = None,
):
    """
    Runs pricing and performance benchmarks for the library's models.
    """
    demo_script = Path("demo.py")
    if not demo_script.exists():
        typer.secho(
            f"Error: '{demo_script}' not found in the current directory.",
            fg=typer.colors.RED,
        )
        typer.echo("Please run this command from the project's root directory.")
        raise typer.Exit(1)

    command = [sys.executable, str(demo_script)]
    if model:
        command.extend(["--model", model])
    if technique:
        command.extend(["--technique", technique])

    subprocess.run(command)
