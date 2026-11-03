from __future__ import annotations

import subprocess
from importlib import resources

import typer

__doc__ = """
CLI command for launching the Streamlit dashboard.
"""


def dashboard():
    """
    Launches the Streamlit dashboard application.
    """
    try:
        home = resources.files("optpricing.dashboard").joinpath("Home.py")
        with resources.as_file(home) as app_path:
            typer.echo(f"Launching Streamlit dashboard from: {app_path}")
            subprocess.run(["streamlit", "run", str(app_path)], check=True)
    except FileNotFoundError:
        typer.secho(
            "Error: 'streamlit' command not found. "
            "Hint: `pip install 'optpricing[ui]'`",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)
