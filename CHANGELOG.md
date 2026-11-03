# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions come from git tags
(`vX.Y.Z`) via setuptools_scm.

## [Unreleased]

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

- SABR is now the four-parameter model of Hagan et al. (2002) on the forward: `alpha` is the initial volatility and the new `nu` the volatility of volatility (previously `alpha` played both roles). The old three-parameter form is still accepted with a `DeprecationWarning` (`nu = alpha`). Monte Carlo simulates the forward dynamics (prices move by O((r−q)T(1−β)) relative to the old spot-with-drift kernel) and the SABR kernels' `alpha` argument is renamed `nu`.
- Heston and Bates default parameters now satisfy the Feller condition (`vol_of_vol` 0.5 → 0.3).

### Added

- `SABRModel.implied_volatility` (Hagan's lognormal expansion, bit-identical to QuantLib's `sabrVolatility`), an optional `shift` parameter for shifted SABR, and a Black-76 closed form so `ClosedFormTechnique`, the CLI and calibration price SABR analytically.
- `benchmarks/run_benchmarks.py` with recorded before/after results, and a reference-value test suite (`tests/correctness/`) pinned to QuantLib 1.44 and textbook closed forms.

## [2.0.0] - 2025-06-26

### Added

- Shipped to PyPI `pyproject.toml`
