# Installation

`optpricing` is designed for a straightforward installation using `pip` and is compatible with Python 3.10 and higher.

## User Installation

Install the latest stable release from PyPI to get:

- The core Python API
- The `optpricing` CLI
- The interactive dashboard

```bash
pip install optpricing
```

Confirm the installation by checking the package version:

```bash
import optpricing
print(optpricing.__version__)  # Prints the installed version
```

Launch the CLI or dashboard:

```bash
optpricing --help  # View CLI commands
optpricing/dashboard  # Launch the dashboard
```

For more details, visit the [Getting Started guide](getting_started.md)

## Developer Installation

To contribute, run tests, or execute benchmarks, follow these steps inside a virtual environment:

```bash
# 1. Clone the repo
git clone https://github.com/diljit22/quantfin.git
cd quantfin

# 2. Create & activate a venv (Linux/macOS)
python -m venv .venv
source .venv/bin/activate  # On Windows, use: .venv\Scripts\activate

# 3. Install editable with dev-extras
pip install --upgrade pip
pip install -e ".[dev]"
```

### Running Benchmarks and Tests

- The benchmark scripts are located in the `examples/` directory and require a local clone.

- Run the test suite with:

    ```bash
    pytest tests/ --cov=src/optpricing --cov-report=term-missing
    ```

    A live pulse is available at <https://app.codecov.io/gh/diljit22/quantfin>.
