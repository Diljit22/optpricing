# optpricing

[![CI/CD](https://github.com/diljit22/optpricing/actions/workflows/ci.yml/badge.svg)](https://github.com/diljit22/optpricing/actions/workflows/ci.yml)
[![PyPI Version](https://badge.fury.io/py/optpricing.svg)](https://pypi.org/project/optPricing/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**A Python library for pricing and calibrating financial options.**

## Introduction

Welcome to **optpricing**, a comprehensive Python toolkit for pricing and calibrating financial derivatives. This library was originally designed for me to learn about the more nuanced methods of quantitative finance and has since grown into a robust framework for analysis.

optpricing is structured around four core pillars:

- **Atoms**: Fundamental data types (`Option`, `Stock`, `Rate`) that ensure consistency and clarity of inputs across the library.
- **Models**: A broad library ranging from classical Black-Scholes-Merton to advanced stochastic volatility (Heston, SABR) and jump/Lévy processes.
- **Techniques**: Multiple pricing engines—closed-form formulas, FFT, numerical integration, PDE solvers, lattice methods, and Monte Carlo (`numba`-accelerated with variance reduction methods baked in).
- **Workflows**: High-level orchestrators that automate end-to-end tasks like daily calibration and out-of-sample backtesting, all accessible via a CLI or an interactive dashboard.

---

## Quick Start

Get started in minutes using the command-line interface.

```bash
# 1. Install the library with all features, including the dashboard
pip install "optPricing"

# 2. Download historical data for a ticker (used by some models)
optpricing data download --ticker SPY

# 3. Launch the interactive dashboard to visualize the results
optpricing dashboard

```

## Documentation & Links

For a detailed tutorial, full API reference, and more examples, please see the official documentation.

- **Getting Started**:  
  [Installation Guide](https://diljit22.github.io/optpricing/guide/installation/) ·  
  [Walkthrough](https://diljit22.github.io/optpricing/guide/getting_started/)

- **Documentation**:  
  [API Reference](https://diljit22.github.io/optpricing)

- **Interactive Dashboard**:  
  [Launch the UI](https://diljit22.github.io/optpricing/guide/dashboard/)

- **About Me**:  
  [LinkedIn](https://www.linkedin.com/in/singhdiljit/)

## Contributing & License

See [CONTRIBUTING](/CONTRIBUTING.md) and [LICENSE](LICENSE).
