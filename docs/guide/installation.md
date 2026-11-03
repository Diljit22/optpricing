# Installation

`optpricing` supports Python 3.11 through 3.14 (3.15 is tested but experimental) and installs with `pip` or `uv`.

## User Installation

Install the latest stable release from PyPI:

```bash
pip install optpricing            # core library + CLI
pip install "optpricing[ui]"      # additionally the Streamlit dashboard
```

The `ui` extra pulls in Streamlit and Plotly, which the core pricing API does not need.

Confirm the installation by checking the package version:

```python
import optpricing

print(optpricing.__version__)  # Prints the installed version
```

Launch the CLI or dashboard:

```bash
optpricing --help      # View CLI commands
optpricing dashboard   # Launch the dashboard (needs the `ui` extra)
```

For more details, visit the [Getting Started guide](getting_started.md)

## Developer Installation

The project uses [uv](https://docs.astral.sh/uv/) for environments and a committed `uv.lock` for reproducible installs.

```bash
# 1. Clone the repo
git clone https://github.com/diljit22/optpricing.git
cd optpricing

# 2. Create the environment and install the project with all dev tooling
#    (Python version comes from .python-version; uv downloads it if needed)
uv sync

# 3. Optional: enable the pre-commit hooks
uv run pre-commit install
```

If you prefer plain `pip` (>= 25.1) inside your own virtual environment:

```bash
pip install -e . --group dev
```

### Running Tests and Benchmarks

- Run the test suite with:

    ```bash
    uv run pytest            # fast, no coverage
    uv run pytest --cov      # with the coverage report CI uses
    ```

- Lint and format with `uv run ruff check .` and `uv run ruff format .`.
- Build the documentation locally with `uv run mkdocs serve`.
- The benchmark scripts are located in the `examples/` directory and require a local clone.

A live coverage report is available at <https://app.codecov.io/gh/diljit22/optpricing>.
