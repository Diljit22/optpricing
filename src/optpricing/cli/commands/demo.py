from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Annotated

import typer

__doc__ = """
CLI command for running benchmark demos from the `examples` directory.
"""


app = typer.Typer(
    name="demo",
    help="Runs benchmark demos from the `examples` directory.",
    add_completion=False,
    no_args_is_help=True,
)

EXAMPLES_DIR = Path("examples")


def _run_example(script_name: str, args: list[str]):
    """Helper function to find and run an example script."""
    script_path = EXAMPLES_DIR / script_name
    if not script_path.exists():
        typer.secho(
            f"Error: Example script '{script_path}' not found.",
            fg=typer.colors.RED,
        )
        typer.echo("Hint: Please run this command from the project's root directory.")
        raise typer.Exit(1)

    command = [sys.executable, str(script_path)] + args
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        typer.secho(
            f"Error running example '{script_name}': Process exited: {e.returncode}",
            fg=typer.colors.RED,
        )
        raise typer.Exit(1)


@app.command(name="european")
def demo_european(
    model: Annotated[
        str | None,
        typer.Option(
            "--model",
            "-m",
            help="Run for a specific model (e.g., 'BSM').",
            case_sensitive=False,
        ),
    ] = None,
    technique: Annotated[
        str | None,
        typer.Option(
            "--technique",
            "-t",
            help="Run for a specific technique (e.g., 'MC').",
            case_sensitive=False,
        ),
    ] = None,
):
    """Runs the European options pricing and performance benchmark."""
    args = []
    if model:
        args.extend(["--model", model])
    if technique:
        args.extend(["--technique", technique])
    _run_example("european_options_benchmark.py", args)


@app.command(name="american")
def demo_american():
    """Runs the American options pricing benchmark."""
    _run_example("american_options_benchmark.py", [])


@app.command(name="rates")
def demo_rates():
    """Runs the interest rate models benchmark."""
    _run_example("rate_models_benchmark.py", [])


demo = app
