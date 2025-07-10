# optpricing

[![Tests](https://img.shields.io/github/actions/workflow/status/diljit22/quantfin/ci.yml?branch=main&label=tests)](https://github.com/diljit22/quantfin/actions/workflows/ci.yml)
[![Coverage](https://codecov.io/gh/diljit22/quantfin/branch/main/graph/badge.svg)](https://codecov.io/gh/diljit22/quantfin)
[![Ruff](https://img.shields.io/github/actions/workflow/status/diljit22/quantfin/ci.yml?branch=main&label=ruff)](https://github.com/diljit22/quantfin/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/github/actions/workflow/status/diljit22/quantfin/ci.yml?branch=main&label=docs)](https://github.com/diljit22/quantfin/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/optpricing.svg)](https://pypi.org/project/optpricing/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**A Python library for pricing and calibrating financial options.**

## Introduction

Welcome to **optpricing**, a comprehensive Python toolkit for pricing and calibrating financial derivatives. This library was originally designed for me to learn about the more nuanced methods of quantitative finance and has since grown into a robust framework for analysis.

optpricing is structured around four core pillars:

- **Atoms**: Fundamental data types (`Option`, `Stock`, `Rate`) that ensure consistency and clarity of inputs across the library.
- **Models**: A broad library ranging from classical Black-Scholes-Merton to advanced stochastic volatility (Heston, SABR) and jump/Levy processes.
- **Techniques**: Multiple pricing engines—closed-form formulas, FFT, numerical integration, PDE solvers, lattice methods, and Monte Carlo (`numba`-accelerated with variance reduction methods baked in).
- **Workflows**: High-level orchestrators that automate end-to-end tasks like daily calibration and out-of-sample backtesting, all accessible via a CLI or an interactive dashboard.

---

## Quick Start

Get started in minutes using the command-line interface.

```bash
# 1. Install the library with all features, including the dashboard
pip install "optpricing"

# 2. Download historical data for a ticker (used by some models)
optpricing data download --ticker SPY

# 3. Launch the interactive dashboard to visualize the results
optpricing dashboard

```

## Documentation & Links

For a detailed tutorial, full API reference, and more examples, please see the official documentation.

- **Getting Started**:
  [Installation Guide](https://diljit22.github.io/quantfin/guide/installation/) ·
  [Walkthrough](https://diljit22.github.io/quantfin/guide/getting_started/)

- **Documentation**:
  [API Reference](https://diljit22.github.io/quantfin)

- **Interactive Dashboard**:
  [Launch the UI](https://diljit22.github.io/quantfin/guide/dashboard/)

- **About Me**:
  [LinkedIn](https://www.linkedin.com/in/singhdiljit/)

## Contributing & License

See [CONTRIBUTING](/CONTRIBUTING.md) and [LICENSE](LICENSE).
