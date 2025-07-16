# Guide: Command-Line Interface (CLI)

The `optpricing` CLI is a powerful tool for performing complex financial analysis directly from your terminal. It provides access to pricing, data management, calibration, and backtesting workflows without requiring any Python scripting.

## Main Commands

These are the top-level commands available directly after `optpricing`.

### `price`

Prices a single option using live market data. This command automatically fetches the underlying spot price, estimates the dividend yield, and calculates an implied risk-free rate from the live option chain before pricing the specified contract.

**Usage:**
`optpricing price [OPTIONS]`

**Key Options:**

- `--ticker, -t TEXT`: Stock ticker (e.g., 'AAPL'). **[required]**
- `--strike, -k FLOAT`: Strike price. **[required]**
- `--maturity, -T TEXT`: Maturity date in YYYY-MM-DD format. **[required]**
- `--type TEXT`: 'call' or 'put'. [default: call]
- `--style TEXT`: 'european' or 'american'. [default: european]
- `--model, -m TEXT`: Model to use (e.g., 'BSM', 'Heston'). [default: BSM]
- `--param TEXT`: Set a model parameter (e.g., `sigma=0.2`). Can be used multiple times.

**Example:**

```bash
# Price an American put on TSLA using the Heston model
optpricing price -t TSLA -k 290 -T 2026-01-16 --style american --model Heston
```

### `calibrate`

Calibrates model parameters to saved market data. This workflow loads a market snapshot, filters for liquid options, and uses an optimization routine to find the model parameters that best fit the observed market prices.

**Usage:**
`optpricing calibrate [OPTIONS]`

**Key Options:**

- `--ticker, -t TEXT`: Ticker for the snapshot to use. **[required]**
- `--model, -m TEXT`: Model to calibrate ('BSM' or 'Merton'). Can be used multiple times. **[required]**
- `--date, -d TEXT`: Snapshot date (YYYY-MM-DD). Defaults to the latest available.

**Example:**

```bash
# Calibrate the BSM and Merton models to a specific historical snapshot
optpricing calibrate -t SPY -m BSM -m Merton --date 2025-07-08
```

### `backtest`

Runs a historical out-of-sample backtest for a model. The workflow iterates through all available historical data, calibrates the model on day `D`, and evaluates its pricing accuracy on the unseen data from day `D+1`.

**Usage:**
`optpricing backtest [OPTIONS]`

**Key Options:**

- `--ticker, -t TEXT`: Ticker to backtest. **[required]**
- `--model, -m TEXT`: Model to backtest. **[required]**
- `--verbose, -v`: Enable detailed logging output.

**Example:**

```bash
optpricing backtest -t SPY -m BSM -v
```

### `dashboard`

Launches the interactive Streamlit dashboard.

**Usage:**
`optpricing dashboard`

---

## Sub-Command Groups

### `data`

Tools for downloading and managing market data.

**Usage:**
`optpricing data [COMMAND]`

**Commands:**

- `download`: Downloads and saves historical log returns for one or more tickers.

  ```bash
  optpricing data download --ticker AAPL --period 5y
  ```

- `snapshot`: Fetches and saves a live market data snapshot of the full option chain.

  ```bash
  optpricing data snapshot --ticker TSLA
  ```

- `dividends`: Fetches and displays live forward dividend yields.

  ```bash
  optpricing data dividends --ticker JPM
  ```

### `demo`

Runs benchmark scripts from the `examples/` directory (requires a local clone of the repository).

**Usage:**
`optpricing demo [COMMAND]`

**Commands:**

- `european`: Runs the European options benchmark across all models.

  ```bash
  optpricing demo european --model Heston
  ```

- `american`: Runs the American options benchmark.

  ```bash
  optpricing demo american
  ```

- `rates`: Runs the interest rate models benchmark.

  ```bash
  optpricing demo rates
  ```

### `tools`

Miscellaneous financial utility tools.

**Usage:**
`optpricing tools [COMMAND]`

**Commands:**

- `implied-rate`: Calculates the implied risk-free rate from a live call-put pair using put-call parity.

  ```bash
  optpricing tools implied-rate --ticker SPY --strike 630 --maturity 2025-12-19
  ```
