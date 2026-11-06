# OptPricing: A Quantitative Finance Library for Derivative Pricing and Analysis

[![CI](https://img.shields.io/github/actions/workflow/status/diljit22/optpricing/ci.yml?branch=main&label=ci)](https://github.com/diljit22/optpricing/actions/workflows/ci.yml)
[![Coverage](https://codecov.io/gh/diljit22/optpricing/branch/main/graph/badge.svg)](https://codecov.io/gh/diljit22/optpricing)
[![Docs](https://img.shields.io/github/actions/workflow/status/diljit22/optpricing/pages.yml?branch=main&label=docs)](https://github.com/diljit22/optpricing/actions/workflows/pages.yml)
[![PyPI version](https://img.shields.io/pypi/v/optpricing.svg)](https://pypi.org/project/optpricing/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

`optpricing` is a Python library for pricing, calibrating, and analyzing financial derivatives. It is built with a focus on architectural clarity, model breadth, and practical usability through a robust API, command-line interface, and an interactive dashboard.

Diljit Singh
[LinkedIn](https://linkedin.com/in/singhdiljit/)

---

## Core Features

* **Model Library**: Implements a comprehensive set of models, including:
  * **Stochastic Volatility**: Heston, SABR
  * **Jump-Diffusion**: Merton, Bates, Kou, SABR with Jumps
  * **Pure Levy Processes**: Variance Gamma (VG), Normal Inverse Gaussian (NIG), CGMY, Hyperbolic
  * **Interest Rate Models**: Vasicek, Cox-Ingersoll-Ross (CIR), Put-Call Parity Implied Rate
  * **Local Volatility**: CEV, Dupire

* **Pricing Engines**: Provides a suite of numerical methods, allowing for technique comparison and validation:
  * Analytic closed-form solutions
  * Characteristic-function pricing: COS (Fang–Oosterlee), numerical integration, and Carr–Madan FFT
  * Finite difference (PDE) solver using a Crank-Nicolson scheme
  * Binomial and trinomial tree methods (CRR, TOPM, Leisen-Reimer) for European and American Black–Scholes options
  * High-performance Monte Carlo engine for European and American options, accelerated with `numba`, with antithetic variates

* **Interfaces**:
  * **Programmatic API**: Use the package as a Python library to build custom financial models in your scripts. Define options, stocks, rates, and models programmatically to compute prices and other metrics.
  * **Command-Line Interface (CLI)**: A robust CLI for live pricing, data management, model calibration, and historical backtesting.
  * **Interactive Dashboard (UI)**: A Streamlit application for visual analysis of option chains, implied volatility surfaces, and model calibrations.

* **Workflow Automation**: High-level classes that orchestrate complex tasks like daily calibration runs and out-of-sample performance evaluation.

---

## Quick Start

`optpricing` installs with `pip` (or `uv`) and supports Python 3.11 through 3.14.

### 1. Install the Library

```bash
pip install optpricing          # library + CLI
pip install "optpricing[ui]"    # also the Streamlit dashboard
```

### 2. Download Historical Data

Some models require historical data (e.g., for calibration). Download data for a ticker like SPY:

```bash
optpricing data download --ticker SPY
```

For more details, see the [Getting Started Guide](https://diljit22.github.io/optpricing/guide/getting_started/).

### 3. Use the CLI

Price an option directly from the terminal. The command below fetches the live option chain for AAPL and prices the contract with the spot, the 13-week T-bill rate, and the dividend yield implied by put-call parity. Heston has no closed form, so the default technique is COS (Fang–Oosterlee):

```bash
optpricing price --ticker AAPL --strike 630 --maturity 2027-06-18 --model Heston --param "rho=-0.7" --param "vol_of_vol=0.5"
```

The same contract as an American option uses Longstaff–Schwartz Monte Carlo. Lattice techniques price American Black–Scholes only:

```bash
optpricing price -t AAPL -k 630 -T 2027-06-18 --style american --model Heston --param "rho=-0.7" --param "vol_of_vol=0.5"
```

For more details, see the [CLI Guide](https://diljit22.github.io/optpricing/guide/CLI/).

### 4. Launch the Dashboard

Visualize option chains and model outputs, and interact with a pricing calculator covering 12 models and 9 techniques (the full library exposes 17 pricing models plus parity/implied-rate models, and 10 techniques, including American Monte Carlo). Requires the `ui` extra.

```bash
optpricing dashboard
```

For more details, see the [Dashboard Guide](https://diljit22.github.io/optpricing/guide/dashboard/).

### 5. Use the Programmatic API

The most powerful way to use the package is via the API, which provides customization of nearly every aspect of pricing:

```python
from optpricing import Option, OptionType, Rate, Stock, ZeroCouponBond
from optpricing.models import BSMModel, CIRModel, VasicekModel
from optpricing.techniques import ClosedFormTechnique

# Define an option, underlying and rate
option = Option(strike=105, maturity=1.0, option_type=OptionType.CALL)
stock = Stock(spot=100, dividend=0.01)
rate = Rate(rate=0.05)

# Choose a model and technique
bsm_model = BSMModel(params={"sigma": 0.20})
cf_technique = ClosedFormTechnique()

result = cf_technique.price(option, stock, bsm_model, rate)
print(f"The option price is: {result.price:.4f}")


delta = cf_technique.delta(option, stock, bsm_model, rate)
gamma = cf_technique.gamma(option, stock, bsm_model, rate)
vega = cf_technique.vega(option, stock, bsm_model, rate)

print(f"Delta: {delta:.4f}")
print(f"Gamma: {gamma:.4f}")
print(f"Vega:  {vega:.4f}")

target_price = 7.50
iv = cf_technique.implied_volatility(
    option, stock, bsm_model, rate, target_price=target_price
)
print(f"Implied volatility for price ${target_price:.2f}: {iv:.4%}")


# Zero Coupon Bond
bond = ZeroCouponBond(maturity=1.0)
r0_stock = Stock(spot=0.05)  # initial short rate
dummy_rate = Rate(rate=0.0)  # ignored by rate models

vasicek = VasicekModel(params={"kappa": 0.86, "theta": 0.09, "sigma": 0.02})
cir = CIRModel(params={"kappa": 0.86, "theta": 0.09, "sigma": 0.02})

p_vasi = cf_technique.price(bond, r0_stock, vasicek, dummy_rate).price
p_cir = cf_technique.price(bond, r0_stock, cir, dummy_rate).price

print(f"Vasicek ZCB Price: {p_vasi:.4f}")
print(f"CIR ZCB Price:     {p_cir:.4f}")
```

For more details, see the [API Guide](https://diljit22.github.io/optpricing/guide/API/).

---

## Documentation

The full documentation includes installation instructions, user guides, examples, and a complete API reference.

* **[View the Official Documentation](https://diljit22.github.io/optpricing/)**

### Guides

* [Introduction](https://diljit22.github.io/optpricing/guide/introduction/)
* [Installation](https://diljit22.github.io/optpricing/guide/installation/)
* [Getting Started](https://diljit22.github.io/optpricing/guide/getting_started/)
* [CLI](https://diljit22.github.io/optpricing/guide/CLI/)
* [Dashboard](https://diljit22.github.io/optpricing/guide/dashboard/)
* [API](https://diljit22.github.io/optpricing/guide/API/)
* [Examples](https://diljit22.github.io/optpricing/guide/examples/)
* [API Reference](https://diljit22.github.io/optpricing/reference/)

## Contributing

Contributions are welcome; see [CONTRIBUTING](/CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
