# Contributing to optpricing

Thank you for your interest in improving optpricing! Even though this is a small project, contributions are welcome—bug reports, documentation fixes, or new feature suggestions.

## Getting Started

1. **Fork** the repository on GitHub.
2. **Clone** your fork locally:

   ```bash
   git clone https://github.com/diljit22/optpricing.git
   cd optpricing
   ```

3. **Install** the project and development tooling with [uv](https://docs.astral.sh/uv/):

   ```bash
   uv sync                      # creates .venv from uv.lock (Python from .python-version)
   uv run pre-commit install    # optional but recommended
   ```

   Without uv, use `pip install -e . --group dev` (pip >= 25.1) in a virtual environment.

## Coding Style

* This project uses **Ruff** for linting and formatting; CI enforces both:

  ```bash
  uv run ruff format .
  uv run ruff check .
  ```

* Please write clear, modular code and include docstrings in NumPy style.

## Testing

Run the full test suite with:

```bash
uv run pytest          # quick
uv run pytest --cov    # with coverage, as CI runs it
```

CI runs the suite on Python 3.11–3.14 (3.15 experimental) and additionally against the minimum supported dependency versions, so keep new dependency floors honest.

## Submitting Changes

1. Create a **feature branch** (`git checkout -b feature/my-change`).
2. Commit your changes with descriptive messages.
3. Push to your fork (`git push origin feature/my-change`).
4. Open a **Pull Request** against `main`.

Please reference any relevant issue numbers in your PR description.

## Reporting Issues and Requests

* For bugs, use the **Bug report** issue template.
* For new features, use the **Feature request** template.

---

For more details, see our [Code of Conduct](CODE_OF_CONDUCT.md) and [Pull Request Template](PULL_REQUEST_TEMPLATE.md).
