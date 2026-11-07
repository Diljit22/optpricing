# Guide: Command-Line Interface (CLI)

The `optpricing` CLI is a powerful tool for performing complex financial analysis directly from your terminal. It provides access to pricing, data management, calibration, backtesting and tuning workflows without requiring any Python scripting.

Everything except `price`, `tools implied-rate` and the `data` download commands works **offline** on stored snapshots: a snapshot carries the spot, the risk-free rate (13-week T-bill) and the dividend yield implied by put-call parity, so no network call is made while calibrating or backtesting.

## Main Commands

These are the top-level commands available directly after `optpricing`.

### `price`

Prices a single option using live market data. The live chain brings its own spot, T-bill rate and parity-implied dividend yield, which are used to price the specified contract.

**Usage:**
`optpricing price [OPTIONS]`

**Key Options:**

- `--ticker, -t TEXT`: Stock ticker (e.g., 'AAPL'). **[required]**
- `--strike, -k FLOAT`: Strike price. **[required]**
- `--maturity, -T TEXT`: Maturity date in YYYY-MM-DD format. **[required]**
- `--type TEXT`: 'call' or 'put'. [default: call]
- `--style TEXT`: 'european' or 'american'. [default: european]
- `--model, -m TEXT`: Model to use (e.g., 'BSM', 'Heston', 'VG'; aliases such as 'Variance Gamma' are accepted). [default: BSM]
- `--technique, -x TEXT`: Force `mc`, `american_mc`, `crr`, or `lr` (`crr` and `lr` price Black–Scholes only). If omitted, a European option uses a closed form, otherwise COS for characteristic-function models (Heston included), otherwise Monte Carlo. American Black–Scholes uses Leisen–Reimer; any other American model uses Longstaff–Schwartz.
- `--param TEXT`: Set a model parameter (e.g., `sigma=0.2`). Can be used multiple times.

**Example:**

```bash
# American Heston uses Longstaff-Schwartz. Trees price Black-Scholes only.
optpricing price -t TSLA -k 290 -T 2027-06-18 --type put --style american --model Heston
```

### `calibrate`

Calibrates model parameters to a saved market snapshot and ranks the models. The workflow selects the liquid options (OTM, tight spreads, open interest, 7 days to 2 years, forward moneyness 0.7–1.3), fits in implied-volatility space with inverse-spread weights from several starting points, evaluates the fit on a wider set and prints a leaderboard sorted by evaluation IV RMSE.

Every model with a closed form or a characteristic function can be calibrated: BSM, Merton, Heston, Bates, SABR, CEV, Kou, NIG, VG, CGMY and Hyperbolic. SABR-with-jumps needs Monte Carlo pricing and is only included with `--allow-mc`.

**Usage:**
`optpricing calibrate [OPTIONS]`

**Key Options:**

- `--ticker, -t TEXT`: Ticker for the snapshot to use. **[required]**
- `--model, -m TEXT`: Model to calibrate, or `all`. Can be used multiple times. [default: all]
- `--date, -d TEXT`: Snapshot date (YYYY-MM-DD). Defaults to the latest available.
- `--fix TEXT`: Hold a parameter fixed (e.g., `sigma=0.25`). Can be used multiple times.
- `--auto`: Use the settings tuned for this ticker by `optpricing tune`.
- `--allow-mc`: Also calibrate simulation-only models (slow).
- `--objective TEXT`: `iv` (default) or `price`.
- `--n-starts INT`: Starting points per model. [default: 4]

Parameters are saved to `artifacts/calibrated_params/{TICKER}_{MODEL}_{DATE}.json` and the leaderboard to `artifacts/calibration_logs/{TICKER}_leaderboard_{DATE}.csv`.

**Example:**

```bash
# Rank every model on the latest SPY snapshot
optpricing calibrate -t SPY

# Calibrate two models to a specific snapshot with the tuned settings
optpricing calibrate -t SPY -m Heston -m Bates --date 2025-07-08 --auto
```

### `backtest`

Runs a walk-forward out-of-sample backtest. Snapshot dates are processed in ascending order: the model is calibrated on day `t` and evaluated, with day `t+1`'s own market parameters, on the liquid options of day `t+1`. Each period also reports two baselines: a flat Black-Scholes volatility at day `t`'s ATM implied vol (`BSM-ATM`) and the parameters calibrated one snapshot earlier (`Stale`).

**Usage:**
`optpricing backtest [OPTIONS]`

**Key Options:**

- `--ticker, -t TEXT`: Ticker to backtest. **[required]**
- `--model, -m TEXT`: Model to backtest, or `all`. Can be used multiple times. [default: all]
- `--auto`: Use the settings tuned for this ticker.
- `--allow-mc`: Also backtest simulation-only models (slow).
- `--n-starts INT`: Starting points per calibration. [default: 2]
- `--verbose, -v`: Enable detailed logging output.

Per model, a per-period CSV, a bucket CSV (errors by maturity and moneyness) and a JSON summary are written to `artifacts/backtest_logs/`.

**Example:**

```bash
optpricing backtest -t SPY -m Merton -m Heston -v
```

### `tune`

Tunes the calibration settings per model by walk-forward cross-validation over the ticker's snapshot corpus (candidate liquidity windows, objectives, weightings and start counts) and ranks the models by mean out-of-sample IV RMSE. The result is cached in `artifacts/tuning/{TICKER}.json` and picked up by `calibrate --auto`, `backtest --auto` and the dashboard.

**Usage:**
`optpricing tune [OPTIONS]`

**Key Options:**

- `--ticker, -t TEXT`: Ticker to tune for (needs at least two snapshots). **[required]**
- `--model, -m TEXT`: Models to include. [default: all]
- `--candidates TEXT`: `quick` (3 settings, default), `full`, or a comma-separated list of candidate names.
- `--max-pairs INT`: Use only the most recent N day pairs.
- `--allow-mc`: Include simulation-only models (slow).

**Example:**

```bash
optpricing tune -t SPY --candidates full
optpricing calibrate -t SPY --auto
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

- `download`: Downloads historical prices, dividends and log returns for one or more tickers. `--rates` also stores the 13-week T-bill history (`^IRX`) used as the risk-free rate for older snapshots.

  ```bash
  optpricing data download --ticker AAPL --period 5y
  optpricing data download --rates
  ```

- `snapshot`: Fetches and saves a live market data snapshot of the option chain. The file carries the spot (`spot_price`), the T-bill rate and the parity-implied dividend yield, which is what calibration and the backtest use. `--compact` keeps only liquid options and compresses the file (~25 KB per ticker and day); `--only-fresh` skips a chain with no trade on the New York session (a holiday, or a clock past midnight UTC that still belongs to the previous evening). When the live book has been cleared but the contract traded in that session, the stored price is the last trade (`price_source` is `last`; otherwise `mid`).

  ```bash
  optpricing data snapshot --ticker TSLA
  optpricing data snapshot --all --compact --only-fresh
  ```

  The scheduled GitHub Action (`.github/workflows/snapshot.yml`) runs at 19:45 UTC, Monday to Friday, for the `snapshot_tickers` in `config.yaml`. It also refreshes `data download` for those tickers (adjusted close, dividends and log returns — the jump-model prior, not the option spot) and the `^IRX` history, and commits `data/` to the `data` branch. Overlay a main checkout with `git fetch origin data && git checkout origin/data -- data/`.

- `stream`: Records live quotes from Yahoo Finance's free streaming endpoint into a parquet file (`data/live_quotes/` by default). Underlyings, indices and option contracts can be mixed; `--contracts N` adds the N most active contracts of each ticker.

  ```bash
  optpricing data stream --ticker SPY --duration 300 --contracts 20
  optpricing data stream --symbol ^SPX --symbol SPY250919C00600000 -d 60
  ```

- `dividends`: Fetches and displays live trailing dividend yields.

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

- `implied-rate`: Calculates the implied risk-free rate from a live call-put pair using put-call parity. The maturity has to be an expiry on that chain; this is separate from the T-bill rate used by `price`.

  ```bash
  optpricing tools implied-rate --ticker SPY --strike 630 --maturity 2027-06-18
  ```
