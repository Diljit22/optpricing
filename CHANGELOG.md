# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions come from git tags
(`vX.Y.Z`) via setuptools_scm.

## [Unreleased]

Phase 5 (verification): every model × technique re-checked against QuantLib 1.44, published tables and an
independent COS pricer; property-based and convergence-order tests; a `NUMBA_DISABLE_JIT=1` coverage job.

### Fixed

- `PDETechnique.delta/gamma` and `IntegrationTechnique.delta` returned Greeks cached by the previous `price`
  call even for a different option, stock, model or rate; the cache is now keyed on the inputs.
- Heston/Bates `variance_scheme="euler"` floored the variance state each step (the biased "absorption" scheme of
  Lord et al. 2010) instead of the documented full truncation; at ξ = 1, T = 3 the bias drops from +2.6 to +0.1.
- Heston and Bates characteristic functions returned NaN at u = −i when ρξ ≥ κ, so `IntegrationTechnique`
  returned NaN prices there; both are now exact at u ∈ {0, −i}.
- `IntegrationTechnique` and the chain pricer truncated the Gil–Pelaez integrals at u = 200, losing up to 6e-3
  at short maturities; the default `upper_bound` is now 1000.
- CRR/Leisen–Reimer/TOPM trees and `AmericanMonteCarloTechnique` silently priced unsupported models (CEV,
  Dupire, Heston on trees) as Black–Scholes; they now raise `TypeError` (the CLI reports it).
- Dupire Monte Carlo crashed for the default model and any plain-Python `vol_surface`; plain scalar
  callables are compiled with numba, uncompilable ones raise a clear `TypeError`.
- CEV closed form near γ = 1 had an O(|γ − 1|) error (4.6e-4) and NaNs from `ncx2` below |γ − 1| = 1e-5; it now
  interpolates to within 1e-7 of QuantLib.
- Vasicek/CIR zero-coupon bonds ignored `face_value`.
- Black's approximation assumed sorted ex-dividend times; model kwargs (`discrete_dividends`,
  `ex_div_times`) were ignored when the `Stock` field existed but was `None`.
- The Hyperbolic calibration recipe fitted the location μ, which the risk-neutral drift makes unidentifiable;
  μ is frozen at 0.
- `IntegrationTechnique` and the chain pricer were under-resolved for slowly decaying characteristic functions
  (VG/CGMY at short maturities: errors up to 1.1; Heston with a strongly violated Feller condition: 0.25, with a
  K = 70 call priced below intrinsic). The chain pricer now uses COS (below); `IntegrationTechnique` checks the
  decay of φ, extends the cut-off and doubles the nodes until two rules agree, and warns when they cannot.
- `FFTTechnique` used a fixed damping α = 1.75 for models without a volatility proxy, which can lie outside
  the moment strip (CGMY, Fang–Oosterlee 2008: 2.8e+99 for a 99.9999 call; Kou η₁ = 1.5: 0.4 for a 37.65
  call). α now follows Lord & Kahl (2007) inside the strip, with puts or C − S e^{−qT} transforms where better.
- `FFTTechnique` stopped the frequency grid at u = 512 and interpolated in log-strike, which left 4.4e-2 at one
  day (σ = 10 %), 6e-4 at T = 0.1 and 2e-5 at T = 1 on Black–Scholes. The grid now extends until φ has decayed
  (up to 2^18 points, with a warning beyond) and is centred on the strike: ≤ 1e-12 vs COS for all CF models
  outside near-degenerate laws, ≤ 5e-10 vs QuantLib for Heston.
- SABR and SABR-with-jumps Monte Carlo used an Euler step floored at zero, which biased the forward upwards
  (β .5, ν .8, T 3: E[F_T] = 100.24 ± 0.04, calls +0.2 vs QuantLib's finite differences). The kernels now use
  the low-bias scheme of Chen, Oosterlee & van der Weide (2012): exact lognormal volatility, then an absorbed
  CEV step drawn exactly (Makarov–Glew Poisson–Gamma mixture). Martingale within 4 SE and calls within
  4 SE + 2e-3 of QuantLib `FdSabrVanillaEngine` on four sets including β .5, ν .8, T 3.

### Changed

- `CEVModel` calls for γ > 1 are now the expectation E[e^{−rT}(S_T − K)^+] (QuantLib `AnalyticCEVEngine`, the
  exact sampler) instead of the Hull/Schroder formula, which includes the strict-local-martingale defect (up
  to 1.98 at γ = 2, T = 2.76). Puts are unchanged; parity reads C − P = e^{−rT} E[S_T] − K e^{−rT}.
- `price_options_vectorized` / `ChainLayout` price characteristic-function models with COS by default
  (`method="cos"`; `"gauss"` keeps the Gauss–Legendre route).
- `select_fastest_technique` (CLI `price` without `--technique`, dashboard IV surfaces, `Calibrator.technique`)
  returns `COSTechnique` instead of `FFTTechnique(n=12)` for characteristic-function models: the same accuracy,
  0.1–0.5 ms per option against 0.4–55 ms.

### Added

- `COSTechnique` (Fang & Oosterlee 2008): puts expanded on a Chernoff-bound truncation range inside the
  model's moment strip, calls by parity, delta from the same series; terms until the series tail is bounded
  by 1e-12 of the strike (1e-9 in the chain pricer), up to 2^16, with a warning when the bound still exceeds
  1e-6. Selectable in the dashboard; FFT, quadrature and the other techniques remain available.
- `BaseModel.moment_strip()`: the open interval of p with E[S_T^p] < ∞ where known in closed form (BSM, Merton,
  Kou, VG, NIG, Hyperbolic, CGMY); used by COS to keep the truncation off poles such as Kou's.
- `IntegrationTechnique(adaptive=True)`: the decay check and node doubling above (`adaptive=False` keeps the
  fixed grid).
- `ParityModel.price_bounds(..., q=0.0)`: no-arbitrage bounds with a dividend yield.
- Property-based tests (`hypothesis` in the `test` group): CF martingale and normalisation, no-arbitrage
  bounds, slopes and convexity in strike, put–call parity, IV round trips for random valid parameters.
- Convergence-order tests (CRR/TOPM order 1, Leisen–Reimer odd N and Crank–Nicolson order 2, Gauss–Legendre
  exponential, Monte Carlo 1/√N) and golden values for Kou, VG, NIG, CGMY, Hyperbolic, CEV γ > 1, Black's
  approximation and the perpetual put.
- CI job running the suite with `NUMBA_DISABLE_JIT=1` so coverage reaches the numba kernels.

### Known limitations

- Near-degenerate laws (VG with T/ν ≪ 1, e.g. one day; CGMY with tiny C·T): φ barely decays, so COS stops at
  2^16 terms and FFT at 2^18 points with errors up to ~1e-3 on S = 100, and Gil–Pelaez quadrature cannot
  converge at all. All three warn.
- `FFTTechnique` on such laws costs 25–55 ms per option (was 0.35 ms, but wrong); COS is the fast route.

## [2.7.0] - 2026-11-04

Workflows: offline data layer, calibration, backtesting, tuning.

### Workflows (calibration, backtesting, data)

- **Calibration rebuilt** (`Calibrator.fit` keeps its signature). Residuals are taken in Black-Scholes
  implied-volatility space (`objective="iv"`, price space optional) with inverse bid-ask-spread weights
  (`"vega"`/`"none"` optional); parameters are optimised in a unit cube (`log_scale` for wide ranges) with
  L-BFGS-B from several starting points (recipe guess, warm start from the previous day, Latin-hypercube
  samples); invalid parameter sets are penalised instead of raising; optional soft Feller penalty; `logging`
  replaces the per-evaluation prints; `Calibrator.result` (`CalibrationResult`) carries loss, IV/price RMSE,
  evaluation counts and per-start records.
- New `LiquidityFilter` / `filter_options` (OTM options with a bid, spread ≤ 25 %, open interest, 7 days–2 years,
  forward moneyness 0.7–1.3, solvable implied vol) and `CalibrationSettings`; fit metrics (`evaluate_model`,
  `residual_table`, `bucket_metrics`) by maturity and moneyness bucket.
- **Every model with a closed form or characteristic function is calibratable** (BSM, Merton, Heston, Bates,
  SABR, CEV, Kou, NIG, VG, CGMY, Hyperbolic); SABR-with-jumps via Monte Carlo behind `--allow-mc`.
  `price_options_vectorized` dispatches to vectorised closed forms (BSM, Merton, SABR, **CEV**), the
  Gauss-Legendre characteristic-function route or Monte Carlo with common random numbers; `ChainLayout` caches
  the strike groups and Fourier kernels for repeated pricing (Heston objective 13 → 4 ms on a 2,700-option chain).
- `CEVModel.price_closed_form`: Cox (1975)/Schroder (1989) noncentral-χ² formula, matching the Hull reference
  values and QuantLib's `AnalyticCEVEngine` to 1e-9 (CEV previously had no closed form).
- `BSMIVSolver`: safeguarded Newton with bracket tracking and bisection fallback, NaN for prices outside the
  no-arbitrage band (deep ITM/OTM and short maturities used to return garbage), optional initial guess.
- **Snapshot schema v2**: `ticker`, `snapshot_date`, `as_of`, `risk_free_rate` (13-week T-bill `^IRX`,
  continuously compounded) and `dividend_yield` (implied from put-call parity near the money, given the rate)
  are stored with every chain; v1 files are upgraded on load from the stored `^IRX` history
  (`optpricing data download --rates`). Workflows no longer call the network.
- Historical files (v2) store adjusted close, dividends and log returns; `trailing_dividend_yield`,
  `load_risk_free_rate`, `get_live_risk_free_rate` added.
- `BacktestWorkflow` is a proper walk-forward test: ascending dates, calibrate on *t*, evaluate on *t+1* with
  that day's market parameters, out-of-sample IV/price RMSE, bucket metrics, `BSM-ATM` and stale-parameter
  baselines, warm starts; CSV + bucket CSV + JSON summary.
- New per-ticker auto-tuning (`optpricing tune`, `workflows.tune_ticker`): walk-forward cross-validation over
  candidate settings, model leaderboard, cache in `artifacts/tuning/{TICKER}.json` used by `calibrate --auto`,
  `backtest --auto` and the dashboard.
- CLI: `calibrate --model all|<names> [--auto] [--allow-mc] [--objective] [--n-starts]` with a leaderboard
  table and CSV, `backtest --model all`, new `tune`, `data download --rates`, `data snapshot --compact
  --only-fresh`, new `data stream` (free `yfinance.WebSocket` quotes → parquet, option contracts included).
  Model names are case-insensitive with aliases (`Variance Gamma` → `VG`).
- Scheduled GitHub Action (`snapshot.yml`) captures compact liquid-only chains for `snapshot_tickers` every
  trading day (~25 KB per ticker) and commits them.
- Dashboard: calibration page offers every calibratable model, tuned settings, objective/start controls, a
  leaderboard sorted by IV RMSE and error-by-bucket tables; `DashboardService` runs on the offline workflow.
  Headless `AppTest` smoke tests cover every page.

### Fixed (workflows)

- Backtest paired dates newest-first, calibrating on *t+1* and evaluating on *t* (look-ahead bias).
- Calibration workflows fetched today's dividend yield from yfinance for historical snapshots (and failed offline).
- Recipes were ignored when a model had `param_defs`; Bates fitted a non-existent `sigma`; Kou's `frozen` was a
  list and crashed; Merton froze all four parameters from history and fitted nothing; Heston started from a
  Feller-violating guess; the VG recipe key (`"Variance Gamma"`) did not match the dashboard's `"VG"`.
- Implied dividend yields: inferring both `r` and `q` from the strike slope of `C − P` is biased by the early
  exercise premium of American puts (real SPY chains implied r ≈ 0 and q ≈ −3 %); the rate now comes from the
  T-bill quote and only near-the-money pairs are used for `q`.
- yfinance 1.x: `fast_info["last_price"]` → `lastPrice` (every live fetch triggered a second `history()` call);
  `dividendYield` is reported in percent since 0.2.5x (unit detection added); duplicate contracts dropped.
- Heston characteristic function at `u = −i` is degenerate when ρξ > κ; the chain pricer falls back to the
  analytic forward there and silences the resulting warnings during calibration.
- `calibrate` printed the parameters at every objective evaluation.
- `pyarrow` (the parquet engine every snapshot needs) was only installed through the `ui` extra; it is now a
  runtime dependency (`pyarrow>=22`, the first release with Python 3.14 wheels).
- `typer>=0.16`: Typer 0.15.x fails on Click ≥ 8.2 (`make_metavar()`), which only the
  minimum-versions CI job exercised.

## [2.6.0] - 2026-11-03

### Performance

- Monte Carlo engine rebuilt: the numba kernels draw their own random numbers from per-chunk
  `numpy.random.Generator`s (spawned from one seed, fixed chunk size) and run concurrently on a thread pool
  (`n_jobs`, `chunk_size` options; `OPTPRICING_MC_THREADS`). Results depend only on the seed and chunk size,
  common-random-number Greeks are preserved, and jump sizes now come from the seeded stream. 100k-path
  BSM 27.7 → 3.4 ms, Heston (Euler) 88.7 → 7.5 ms on 10 cores.
- Heston/Bates Monte Carlo defaults to Andersen's Quadratic-Exponential variance scheme with martingale
  correction (`variance_scheme="qe"`; `"euler"` keeps full truncation). 20 QE steps are more accurate than
  252 Euler steps; on long-dated, high vol-of-vol parameters Euler was off by several price units.
- Closed forms use `scipy.special.ndtr` (BSM price 28 → 2 µs); the Merton series is vectorised with log-space
  Poisson weights (2.7 ms → 14 µs, no overflow past 170 terms).
- `IntegrationTechnique` evaluates the Gil–Pelaez integrals on a cached composite Gauss–Legendre rule
  (`method="gauss"`, 0.69 → 0.07 ms; `method="quad"` keeps the adaptive rule); the calibration chain pricer
  uses the same rule (400 options 19.5 → 1.7 ms). `FFTTechnique` interpolates cubically and defaults to
  `eta=0.125` (ATM error 5e-4 → 1e-6 at the same cost).
- Longstaff–Schwartz: parallel power-sum regression on moneyness, time-major path matrices
  (20k paths 32 → 13 ms); Crank–Nicolson PDE marches in numba with the two constant matrices factorised once
  (200×200 grid 2.3 → 0.3 ms).
- Kernel signatures changed: SDE kernels take `(gen, n_paths, n_steps, antithetic, ...)`, path kernels return
  `(n_steps + 1, n_paths)` arrays, and `longstaff_schwartz_pricer` takes `n_threads`.

## [2.5.0] - 2026-11-03

Correctness audit against QuantLib 1.44 and the literature.

### Fixed

- **European Monte Carlo for Heston, Bates, SABR and SABR-jump applied the spot/variance correlation twice** (once in `MonteCarloTechnique`, once in the kernels), biasing prices by up to 2 % at ρ = −0.7. Heston MC now agrees with QuantLib's analytic engine within Monte Carlo error.
- NIG exact sampler drew the inverse-Gaussian subordinator with shape 1 instead of (δT)²; prices were ~3 % low.
- CEV exact sampler used an incorrect noncentral-χ² parametrisation (terminal mean 100.6 for a forward of 105.1), ignored the dividend yield and the seeded generator, and divided by zero at r = 0. It now samples the absorbed squared-Bessel law exactly (Makarov & Glew 2010) and matches the Cox/Schroder closed form.
- CGMY: the Lévy exponent was singular at Y = 0 and Y = 1 (Γ(−Y) poles) and the only Monte Carlo path was guarded on Y = 1 while implementing the Y = 0 law. Both limits are closed-form now; other Y sample by characteristic-function inversion.
- Hyperbolic model: characteristic function was neither the Lévy-process law nor the horizon-T hyperbolic law (so the drift was not a martingale correction for T ≠ 1), and the sampler passed α, β where scipy expects αδ, βδ. It is now the hyperbolic Lévy motion (φₜ = φ₁ᵗ) with exact sampling at T = 1 and CF inversion otherwise.
- PDE (Crank–Nicolson) Greeks were read at the grid node below the spot (delta off by ≈ 0.02); the spot now sits on a node, boundary values track the time level (with the dividend discount), and a Rannacher start-up is used.
- Longstaff–Schwartz regressed on raw powers of S through an explicit inverse of XᵀX (ill-conditioned beyond degree 2) and swallowed failures with a bare `except`; it now uses moneyness-scaled regressors and least squares.
- SABR kernels could leave the spot negative after a large down-move (puts paid more than the strike); zero is absorbing for 0 < β < 1.
- Parameter validation now rejects parameter sets without a risk-neutral measure: Kou η₁ ≤ 1, VG θν + σ²ν/2 ≥ 1, NIG/Hyperbolic |β+1| ≥ α, CGMY M ≤ 1.
- `optpricing.config` no longer writes `config.yaml` and data directories at import time (previously into `site-packages/lib` for installed packages); the root is `$OPTPRICING_HOME` or the working directory and folders are created when first written to.
- Heston and Bates read `v0` from the model parameters when it is not passed as a pricing kwarg.
- Feller-condition messages are `UserWarning`s instead of prints; pandas 3 (`inplace`, `Timestamp.utcnow`) and `importlib.resources.path` deprecations removed; `fastmath` dropped from the numba kernels (≤ 5 % measured cost, bit-identical results).

### Changed

- SABR is now the four-parameter model of Hagan et al. (2002) on the forward: `alpha` is the initial volatility and the new `nu` the volatility of volatility (previously `alpha` played both roles). The old three-parameter form is still accepted with a `DeprecationWarning` (`nu = alpha`). Monte Carlo simulates the forward dynamics (prices move by O((r−q)T(1−β)) relative to the old spot-with-drift kernel) and the SABR kernels' `alpha` argument is renamed `nu`.
- Heston and Bates default parameters now satisfy the Feller condition (`vol_of_vol` 0.5 → 0.3).

### Added

- `SABRModel.implied_volatility` (Hagan's lognormal expansion, bit-identical to QuantLib's `sabrVolatility`), an optional `shift` parameter for shifted SABR, and a Black-76 closed form so `ClosedFormTechnique`, the CLI and calibration price SABR analytically.
- `benchmarks/run_benchmarks.py` with recorded before/after results, and a reference-value test suite (`tests/correctness/`) pinned to QuantLib 1.44 and textbook closed forms.

## [2.4.0] - 2026-11-03

Packaging and CI.

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
