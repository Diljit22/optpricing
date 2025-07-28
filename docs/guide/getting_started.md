# Getting Started: A Walkthrough

This guide walks you through a typical use case, from pricing a single option with the Python API to running a full calibration workflow with the command-line interface (CLI).

---

## 1. Pricing a European Option with the Python API

The core components of `optpricing` are designed to be intuitive and composable. Let’s price a standard European call option and calculate its Greeks.

First, define the core **Atoms**: the option contract, the underlying stock, and the risk-free rate.

```python
from optpricing import Option, OptionType, Rate, Stock

# Define the core components
option = Option(strike=105, maturity=1.0, option_type=OptionType.CALL)
stock = Stock(spot=100.0, dividend=0.01)
rate = Rate(rate=0.05)
```

Next, select a **Model** and a pricing **Technique**. Here, we’ll use the Black-Scholes-Merton model and its analytic closed-form solution.

```python
from optpricing.models import BSMModel
from optpricing.techniques import ClosedFormTechnique

# Choose a model and technique
bsm_model = BSMModel(params={"sigma": 0.20})
cf_technique = ClosedFormTechnique()

# Calculate the price
result = cf_technique.price(option, stock, bsm_model, rate)
print(f"The option price is: {result.price:.4f}")

# Calculate Greeks
delta = cf_technique.delta(option, stock, bsm_model, rate)
gamma = cf_technique.gamma(option, stock, bsm_model, rate)
vega = cf_technique.vega(option, stock, bsm_model, rate)

print(f"Delta: {delta:.4f}")
print(f"Gamma: {gamma:.4f}")
print(f"Vega:  {vega:.4f}")
```

---

## 2. Pricing an American Option with the Python API

The API supports American options using models like Heston with Monte Carlo techniques, optimized with `numba` for performance.

```python
from optpricing.models import HestonModel
from optpricing.techniques import AmericanMonteCarloTechnique

# Define components
option = Option(strike=105, maturity=1.0, option_type=OptionType.CALL, style="american")
stock = Stock(spot=100.0, dividend=0.01)
rate = Rate(rate=0.05)

# Choose a model and technique
heston_model = HestonModel(params={"v0": 0.04, "kappa": 2.0, "theta": 0.05, "rho": -0.7, "vol_of_vol": 0.5})
mc_technique = AmericanMonteCarloTechnique()

# Calculate the price
result = mc_technique.price(option, stock, heston_model, rate)
print(f"The American option price is: {result.price:.4f}")
```

---

## 3. Using the Command-Line Interface (CLI)

The CLI provides a powerful way to access the library’s features without writing Python code.

### Pricing an Option

Price a European or American option directly from the terminal. The command below fetches the live option chain for AAPL, retrieves the current dividend rate, calculates the implied risk-free rate from at-the-money contracts, and prices the contract with Heston’s model using its default technique (FFT):

```bash
optpricing price --ticker AAPL --strike 210 --maturity 2025-12-19 --type call --model Heston --param "rho=-0.7" --param "vol_of_vol=0.5"
```

For an American option:

```bash
optpricing price --ticker AAPL --strike 210 --maturity 2025-12-19 --type call --style american --model Heston --param "rho=-0.7" --param "vol_of_vol=0.5"
```

### Downloading Data

Download historical returns or a live option chain snapshot for calibration or backtesting:

```bash
# Download historical log-returns
optpricing data download --ticker SPY --period 10y

# Save a snapshot of the live option chain
optpricing data snapshot --ticker SPY
```

Data is saved to the `data/` directory for use in other workflows.

### Calibrating a Model

Calibrate a model to fit observed market prices using a saved market snapshot:

```bash
# Calibrate the Heston model to the latest snapshot for SPY
optpricing calibrate --ticker SPY --model Heston --verbose
```

This command runs the calibration workflow, prints the results, and saves optimized parameters to a `.json` file in the `artifacts/` directory.

---

## 4. Launching the Dashboard

Visualize option chains and model outputs with an interactive Streamlit dashboard supporting 15 models and 10 techniques:

```bash
optpricing dashboard
```

---

## What’s Next?

Explore the full capabilities of `optpricing` with these guides:

* [Dashboard Guide](https://diljit22.github.io/optpricing/guide/dashboard.md): A visual tour of the interactive UI.
* [Examples Guide](https://diljit22.github.io/optpricing/guide/examples.md): Advanced benchmarks and use cases.
* [API Guide](https://diljit22.github.io/optpricing/guide/API.md): Detailed API documentation for custom workflows.
