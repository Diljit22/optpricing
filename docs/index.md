# OptPricing: A Quantitative Finance Library for Derivative Pricing and Analysis

`optpricing` is a Python library for pricing, calibrating, and analyzing financial derivatives. It is built with a focus on architectural clarity, model breadth, and practical usability through a robust API, command-line interface, and an interactive dashboard.

Diljit Singh
linkedin.com/in/singhdiljit/

---

## Core Features

* **Model Library**: Implements a comprehensive set of models, including:
  * **Stochastic Volatility**: Heston, SABR
  * **Jump-Diffusion**: Merton, Bates, Kou, SABR with Jumps
  * **Pure Levy Processes**: Variance Gamma (VG), Normal Inverse Gaussian (NIG), CGMY, Hyperbolic
  * **Interest Rate Models**: Vasicek, Cox-Ingersoll-Ross (CIR), Put-Call Parity Implied Rate
  * **Local Volatility**: Dupire's Equation

* **Pricing Engines**: Provides a suite of numerical methods, allowing for technique comparison and validation:
  * Analytic closed-form solutions
  * Numerical integration and FFT-based pricing via characteristic functions
  * Finite difference (PDE) solver using a Crank-Nicolson scheme
  * Binomial and trinomial tree methods (CRR, TOPM, Leisen-Reimer) for European and American options
  * High-performance Monte Carlo engine for European and American options, accelerated with `numba`, featuring variance reduction techniques (e.g., antithetic variates, control variates, importance sampling)

* **Interfaces**:
  * **Programmatic API**: Use the package as a Python library to build custom financial models in your scripts. Define options, stocks, rates, and models programmatically to compute prices and other metrics.
  * **Command-Line Interface (CLI)**: A robust CLI for live pricing, data management, model calibration, and historical backtesting.
  * **Interactive Dashboard (UI)**: A Streamlit application for visual analysis of option chains, implied volatility surfaces, and model calibrations.

* **Workflow Automation**: High-level classes that orchestrate complex tasks like daily calibration runs and out-of-sample performance evaluation.

---

## Quick Start

`optpricing` is designed for a straightforward installation using `pip` and is compatible with Python 3.10 and higher.

### 1. Install the Library

```bash
pip install optpricing
```

### 2. Download Historical Data

Some models require historical data (e.g., for calibration). Download data for a ticker like SPY:

```bash
optpricing data download --ticker SPY
```

For more details, see the [Getting Started Guide]((guide/getting_started.md)).

### 3. Use the CLI

Price an option directly from the terminal. The command below fetches the live option chain for AAPL, retrieves the current dividend rate, calculates the implied risk-free rate from at-the-money contracts, and prices the contract with Heston’s model using its default pricing technique (FFT):

```bash
optpricing price --ticker AAPL --strike 630 --maturity 2025-12-19 --model Heston --param "rho=-0.7" --param "vol_of_vol=0.5"
```

To price the same contract as an American Option use:

```bash
optpricing price -t AAPL -k 210 -T 2025-12-19 --style american --model Heston --param "rho=-0.7" --param "vol_of_vol=0.5"
```

For more details, see the [CLI Guide]((guide/CLI.md)).

### 4. Launch the Dashboard

Visualize option chains and model outputs, interact with a pricing calculator featuring 15 models and 10 techniques.

```bash
optpricing dashboard
```

For more details, see the [Dashboard Guide]((guide/dashboard.md)).

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

For more details, see the [API Guide]((guide/API.md)).

---

## Documentation

The full documentation includes installation instructions, user guides, examples, and a complete API reference.

* **[View the Official Documentation](https://diljit22.github.io/quantfin/)**

### Guides

* [Introduction](guide/introduction.md)
* [Installation](guide/installation.md)
* [Getting Started](guide/getting_started.md)
* [CLI](guide/CLI.md)
* [Dashboard](guide/dashboard.md)
* [API](guide/API.md)
* [Examples](guide/examples.md)
* [API Reference](reference/index.md)
