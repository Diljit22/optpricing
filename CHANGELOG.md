# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions come from git tags
(`vX.Y.Z`) via setuptools_scm.

## [Unreleased]

### Changed

- Python 3.10 support dropped (end of life); supported versions are 3.11–3.14, with 3.15 tested as experimental.
- Dependency pins relaxed from `~=` to verified minimum versions (`numpy>=2.3.2`, `scipy>=1.16.1`, `pandas>=2.3.3`, `numba>=0.63`, `pyyaml>=6.0.3`, `typer>=0.15.3`, `yfinance>=0.2.63`). Current releases (numpy 2.5, pandas 3.1, numba 0.68, yfinance 1.x) are supported.
- Streamlit and Plotly moved to the optional `ui` extra: `pip install "optpricing[ui]"` for the dashboard.
- Development tooling moved from the `dev` extra to PEP 735 dependency groups (`uv sync`; `pip install -e . --group dev`).
- Package metadata uses a PEP 639 license expression; the source distribution no longer bundles `data/`, `artifacts/` or docs (24 MB → ~0.1 MB).
- CI rebuilt on uv: lint, Python 3.11–3.15 matrix, minimum-dependency-versions job, build/twine checks; docs and publish workflows split into build and deploy jobs with hash-pinned actions.

### Removed

- Unused runtime dependencies `matplotlib`, `tabulate` and the explicit `click` pin (Click arrives via Typer).

### Fixed

- `test_price_command` relied on a hard-coded maturity date that is now in the past.

## [2.0.0] - 2025-06-26

### Added

- Shipped to PyPI `pyproject.toml`
