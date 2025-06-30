# Getting Started: A Walkthrough

This guide is a hands-on tutorial that will walk you through the library's core features,
from pricing a single option to running a full model calibration, using the command-line
interface (CLI).

---

## 1. Run the Built-In Demo

The quickest way to see optpricing in action is with the included benchmark script.

**Prerequisite**: install the library if you haven’t already:

```bash
pip install optpricing
```

Then execute:

```bash
optpricing demo
```

This will price a standard option across multiple models and techniques and print a
formatted comparison table.

---

## 2. Pricing an Option Programmatically

First, import and configure the objects you need:

```python
from optpricing.atoms import Option, Stock, Rate, OptionType, ZeroCouponBond
from optpricing.models import BSMModel, VasicekModel, CIRModel
from optpricing.techniques import ClosedFormTechnique

# 1. Define an option, underlying and rate
option = Option(strike=105, maturity=1.0, option_type=OptionType.CALL)
stock  = Stock(spot=100, dividend=0.01)
rate   = Rate(rate=0.05)

# 2. Choose a model and technique
bsm_model     = BSMModel(params={"sigma": 0.20})
cf_technique  = ClosedFormTechnique()
```

### 2.1 Compute the Price

```python
result = cf_technique.price(option, stock, bsm_model, rate)
print(f"The option price is: {result.price:.4f}")
>>> The option price is: 7.4917
```

### 2.2 Compute the Greeks

```python
delta = cf_technique.delta(option, stock, bsm_model, rate)
gamma = cf_technique.gamma(option, stock, bsm_model, rate)
vega  = cf_technique.vega(option, stock, bsm_model, rate)

print(f"Delta: {delta:.4f}")
print(f"Gamma: {gamma:.4f}")
print(f"Vega:  {vega:.4f}")

>>> Delta: 0.5172
>>> Gamma: 0.0197
>>> Vega:  39.4353
```

### 2.3 Find Implied Volatility

```python
target_price = 7.50
iv = cf_technique.implied_volatility(
    option, stock, bsm_model, rate, target_price=target_price
)
print(f"Implied volatility for price ${target_price:.2f}: {iv:.4%}")
>>> Implied volatility for price $7.50: 20.0211%
```

---

## 3. Pricing Interest‐Rate Instruments

optpricing reuses the same pricing interfaces for interest‐rate models:

```python
# Zero‐coupon bond maturing in 1 year
bond      = ZeroCouponBond(maturity=1.0)
r0_stock  = Stock(spot=0.05)    # initial short rate
dummy_rate = Rate(rate=0.0)     # ignored by rate models

vasicek = VasicekModel(params={"kappa":0.86, "theta":0.09, "sigma":0.02})
cir     = CIRModel(params={"kappa":0.86, "theta":0.09, "sigma":0.02})

p_vasi = cf_technique.price(bond, r0_stock, vasicek, dummy_rate).price
p_cir  = cf_technique.price(bond, r0_stock, cir,     dummy_rate).price

print(f"Vasicek ZCB Price: {p_vasi:.4f}")
print(f"CIR ZCB Price:     {p_cir:.4f}")

>>> Vasicek ZCB Price: 0.9388
>>> CIR ZCB Price:     0.9388
```

---

## 4. Live Pricing and Analysis via CLI

### 4.1 Price an Option

```bash
optpricing price \
  --ticker SPY \
  --strike 500 \
  --maturity 2025-12-19 \
  --type call \
  --model Heston \
  --param "v0=0.04" \
  --param "kappa=2.0" \
  --param "theta=0.05" \
  --param "rho=-0.7" \
  --param "vol_of_vol=0.5"
```

### 4.2 Implied Rate from Put-Call Parity

The tools implied-rate command fetches live prices for a call-put pair and calculates the risk-free rate implied by put-call parity.

```bash
optpricing tools implied-rate --ticker SPY --strike 500 --maturity 2025-12-19
```

---

## 5. Model Calibration with the CLI

Download historical returns (for initial guesses and jump paramaters):

```bash
optpricing data download --ticker SPY
```

Download a snapshot of the market-data

```bash
# For the 25 benchmark stocks use --all
optpricing data snapshot --all

# If just a particular ticker use e.g.
optpricing data snapshot --ticker SPY --ticker AAPL
```

A simple calibration for the Merton Jump model. The workflow will find the latest market data
for SPY and solve for the implied volatility that best fits the front-month options.

The `--verbose` flag provides detailed logs from the workflow.

```bash
optpricing calibrate --ticker SPY --model Merton --verbose
```

The final calibrated parameters are printed to the console and saved to a JSON file in
the `artifacts/calibrated_params/` directory.

---

## 6. Managing Data

* Download specific tickers:

  ```bash
  optpricing data download --ticker AAPL --ticker TSLA
  ```

* Download all defaults (from `config.yaml`):

  ```bash
  optpricing data download --all
  ```

* Snapshot the live option chain:

  ```bash
  optpricing data snapshot --ticker NVDA
  ```

---

## 7. Running Tests

If you have installed dev dependencies:

```bash
pip install -e .[dev]
pytest
```

---

## 8. Launching the Dashboard

Make sure you have the `[app]` extras installed:

```bash
pip install optpricing[app]
```

Then run:

```bash
optpricing dashboard
```

This will open the Streamlit application in your browser for interactive exploration.
