# OptPricing

[![CI](https://img.shields.io/github/actions/workflow/status/diljit22/optpricing/ci.yml?branch=main&label=ci)](https://github.com/diljit22/optpricing/actions/workflows/ci.yml)
[![Coverage](https://codecov.io/gh/diljit22/optpricing/branch/main/graph/badge.svg)](https://codecov.io/gh/diljit22/optpricing)
[![Docs](https://img.shields.io/github/actions/workflow/status/diljit22/optpricing/pages.yml?branch=main&label=docs)](https://github.com/diljit22/optpricing/actions/workflows/pages.yml)
[![PyPI version](https://img.shields.io/pypi/v/optpricing.svg)](https://pypi.org/project/optpricing/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**[Live dashboard](https://optpricing.streamlit.app/)** · **[Documentation](https://diljit22.github.io/optpricing/)**

Python library for pricing, calibrating, and analyzing derivatives. 17 pricing models and 10 techniques. On the contracts in the validation table, Heston's absolute price error against QuantLib 1.44 is at most 5e-10, and a 400-option Heston chain prices in 1.57 ms.

Diljit Singh · [LinkedIn](https://linkedin.com/in/singhdiljit/)

![OptPricing dashboard](docs/images/optPricing_demo.gif)

The model is the distribution and the technique is the algorithm. The call is always `technique.price(option, stock, model, rate)`. Heston has no closed form, so the default technique is COS (Fang–Oosterlee). The same model also prices by Carr–Madan FFT and by Andersen's quadratic-exponential Monte Carlo. Design notes are in [ARCHITECTURE.md](ARCHITECTURE.md). The formulas are in [Methods](docs/guide/methods.md).

---

## Quick start

Python 3.11–3.14. The block below does not use the network.

```bash
pip install optpricing
```

```python
from optpricing import (
    BSMModel,
    COSTechnique,
    ClosedFormTechnique,
    FFTTechnique,
    HestonModel,
    MonteCarloTechnique,
    Option,
    OptionType,
    Rate,
    Stock,
)

option = Option(strike=100.0, maturity=1.0, option_type=OptionType.CALL)
stock = Stock(spot=100.0, dividend=0.01)
rate = Rate(rate=0.05)

bsm = BSMModel(params={"sigma": 0.20})
analytic = ClosedFormTechnique()
print(f"BSM price {analytic.price(option, stock, bsm, rate).price:.4f}")
print(f"delta    {analytic.delta(option, stock, bsm, rate):.4f}")

heston = HestonModel(
    params={
        "v0": 0.04,
        "kappa": 2.0,
        "theta": 0.05,
        "rho": -0.7,
        "vol_of_vol": 0.3,
    }
)
cos = COSTechnique().price(option, stock, heston, rate).price
fft = FFTTechnique().price(option, stock, heston, rate).price
mc_engine = MonteCarloTechnique(n_paths=50_000, n_steps=100, seed=1)
mc = mc_engine.price(option, stock, heston, rate).price
print(f"Heston  COS {cos:.4f}   FFT {fft:.4f}   MC {mc:.4f}")
print(f"COS - FFT = {cos - fft:.2e}")
```

```
BSM price 9.8263
delta    0.6118
Heston  COS 10.2295   FFT 10.2295   MC 10.2258
COS - FFT = 1.78e-15
```

The Monte Carlo line is fixed by `seed=1`. Delta, gamma, vega, and implied volatility for a Black–Scholes contract are in [Getting started](docs/guide/getting_started.md).

---

## Validation

Maximum absolute error on the contracts in `tests/correctness/`. The reference is a QuantLib 1.44 value pinned in those tests, or a closed form. Heston is 5e-10 against `AnalyticHestonEngine` (7e-10 on the vectorised chain). The largest closed-form gap is CEV near γ = 1, at 6e-8, where the test allows 1e-7. Reprint the table with `uv run python benchmarks/accuracy.py`. The copy with the worst case of each row is in [Validation](docs/guide/validation.md).

<!-- accuracy-summary -->
| Model | Technique | Reference | Cases | Max abs error |
| --- | --- | --- | ---: | ---: |
| Heston | COS | QuantLib 1.44 AnalyticHestonEngine | 18 | 5.0e-10 |
| Heston | Gauss-Legendre | QuantLib 1.44 AnalyticHestonEngine | 18 | 5.0e-10 |
| Heston | FFT | QuantLib 1.44 AnalyticHestonEngine | 5 | 5.0e-10 |
| Heston | chain COS | QuantLib 1.44 AnalyticHestonEngine | 4 | 7.0e-10 |
| Heston | chain Gauss-Legendre | QuantLib 1.44 AnalyticHestonEngine | 4 | 4.1e-11 |
| NIG | COS | COS, N=2^18 (Fang and Oosterlee truncation) | 3 | 8.2e-11 |
| NIG | Gauss-Legendre | COS, N=2^18 (Fang and Oosterlee truncation) | 3 | 8.2e-11 |
| NIG | chain COS | COS, N=2^18 (Fang and Oosterlee truncation) | 3 | 5.2e-10 |
| NIG | chain Gauss-Legendre | COS, N=2^18 (Fang and Oosterlee truncation) | 3 | 8.2e-11 |
| Hyperbolic | COS | COS, N=2^18 (Fang and Oosterlee truncation) | 1 | 1.6e-11 |
| Hyperbolic | Gauss-Legendre | COS, N=2^18 (Fang and Oosterlee truncation) | 1 | 1.6e-11 |
| Hyperbolic | chain COS | COS, N=2^18 (Fang and Oosterlee truncation) | 1 | 3.5e-11 |
| Hyperbolic | chain Gauss-Legendre | COS, N=2^18 (Fang and Oosterlee truncation) | 1 | 1.6e-11 |
| Kou | COS | COS, N=2^18, on the paper characteristic function | 3 | 2.3e-11 |
| Kou | Gauss-Legendre | COS, N=2^18, on the paper characteristic function | 3 | 2.3e-11 |
| NIG | COS | COS, N=2^18, on the paper characteristic function | 3 | 4.5e-11 |
| NIG | Gauss-Legendre | COS, N=2^18, on the paper characteristic function | 3 | 4.5e-11 |
| Hyperbolic | COS | COS, N=2^18, on the paper characteristic function | 3 | 4.3e-11 |
| Hyperbolic | Gauss-Legendre | COS, N=2^18, on the paper characteristic function | 3 | 4.3e-11 |
| VG | COS | Fang and Oosterlee (2008) | 1 | 2.0e-10 |
| VG | Gauss-Legendre | Fang and Oosterlee (2008) | 1 | 2.0e-10 |
| CGMY | COS | Fang and Oosterlee (2008) | 2 | 4.8e-10 |
| CGMY | Gauss-Legendre | Fang and Oosterlee (2008) | 2 | 4.8e-10 |
| Kou | FFT | COS, N=2^18, on the paper characteristic function | 6 | 2.7e-11 |
| CGMY | FFT | COS, N=2^18, on the paper characteristic function | 4 | 5.3e-10 |
| NIG | FFT | COS, N=2^18, on the paper characteristic function | 3 | 4.5e-11 |
| VG | COS | COS, N=2^22, fixed range | 3 | 1.1e-07 |
| CGMY | COS | COS, N=2^22, fixed range | 3 | 5.3e-07 |
| Bates | COS | QuantLib 1.44 BatesEngine | 6 | 4.4e-11 |
| Bates | Gauss-Legendre | QuantLib 1.44 BatesEngine | 8 | 3.6e-10 |
| Merton | closed form | Merton (1976) series, 80 terms | 2 | 4.6e-10 |
| Merton | Gauss-Legendre | Merton (1976) series, 80 terms | 2 | 4.8e-10 |
| Merton | closed form | Gil-Pelaez quadrature | 1 | 1.0e-11 |
| Vasicek | closed form | QuantLib 1.44 discount bond | 4 | 4.9e-11 |
| CIR | closed form | QuantLib 1.44 discount bond | 4 | 4.1e-11 |
| Black's approximation | closed form | Hull, Black's approximation | 4 | 4.8e-11 |
| Perpetual put | closed form | Merton (1973) perpetual put | 8 | 7.1e-15 |
| CEV | closed form | Hull CEV formula (zero rates also QuantLib 1.44) | 3 | 4.2e-10 |
| CEV | Monte Carlo | Hull CEV formula (zero rates also QuantLib 1.44) | 3 | 8.6e-03 |
| CEV | closed form | QuantLib 1.44 AnalyticCEVEngine | 27 | 6.0e-08 |
| BSM | CRR | BSM closed form | 1 | 2.0e-03 |
| BSM | Leisen-Reimer | BSM closed form | 1 | 1.5e-06 |
| BSM | TOPM | BSM closed form | 1 | 9.8e-04 |
| BSM | Crank-Nicolson | BSM closed form | 1 | 5.4e-03 |
| BSM American | CRR | QuantLib 1.44 FdBlackScholesVanillaEngine | 2 | 4.8e-04 |
| BSM American | Leisen-Reimer | QuantLib 1.44 FdBlackScholesVanillaEngine | 2 | 2.9e-04 |
| BSM American | TOPM | QuantLib 1.44 FdBlackScholesVanillaEngine | 2 | 5.6e-04 |
| BSM American | LSMC | QuantLib 1.44 FdBlackScholesVanillaEngine | 2 | 2.1e-02 |
| BSM | FFT | COSTechnique | 3 | 1.8e-13 |
| BSM | Gauss-Legendre | COSTechnique | 3 | 4.3e-14 |
| BSM | COS | BSM closed form | 3 | 6.4e-14 |
| Heston | FFT | COSTechnique | 3 | 7.8e-14 |
| Heston | Gauss-Legendre | COSTechnique | 3 | 4.3e-14 |
| Merton | FFT | COSTechnique | 3 | 1.6e-13 |
| Merton | Gauss-Legendre | COSTechnique | 3 | 6.4e-14 |
| Bates | FFT | COSTechnique | 3 | 1.2e-13 |
| Bates | Gauss-Legendre | COSTechnique | 3 | 4.3e-14 |
| Kou | FFT | COSTechnique | 3 | 1.2e-10 |
| Kou | Gauss-Legendre | COSTechnique | 3 | 1.4e-14 |
| VG | FFT | COSTechnique | 3 | 1.5e-13 |
| VG | Gauss-Legendre | COSTechnique | 3 | 6.2e-08 |
| NIG | FFT | COSTechnique | 3 | 1.0e-13 |
| NIG | Gauss-Legendre | COSTechnique | 3 | 7.1e-14 |
| CGMY | FFT | COSTechnique | 3 | 1.7e-13 |
| CGMY | Gauss-Legendre | COSTechnique | 3 | 4.3e-14 |
| Hyperbolic | FFT | COSTechnique | 3 | 1.8e-13 |
| Hyperbolic | Gauss-Legendre | COSTechnique | 3 | 6.4e-14 |
<!-- /accuracy-summary -->

<!-- accuracy-convergence -->
| Scheme | Resolutions | Measured slopes | Mean | Theory |
| --- | --- | --- | ---: | ---: |
| CRR | 200, 400, 800 | -1.00, -1.00 | -1.00 | -1.00 |
| TOPM | 200, 400, 800 | -1.00, -1.00 | -1.00 | -1.00 |
| Leisen-Reimer | 101, 201, 401 | -1.99, -1.99 | -1.99 | -2.00 |
| Crank-Nicolson | 100, 200, 400 | -2.05, -1.98 | -2.01 | -2.00 |
| Monte Carlo | 2,048 and 32,768 paths | -0.45 | -0.45 | -0.50 |
<!-- /accuracy-convergence -->

Trees and Monte Carlo are in the table on purpose: an American CRR price is not a 1e-9 result, and the Monte Carlo slope is the standard-deviation slope, not a price error.

---

## Performance

Warm start: the numba compile is not in the median. Unless a row says otherwise, the contract is a spot of 100, a strike of 100, one year, a rate of 5%, and a dividend yield of 1%. Black–Scholes volatility is 20%. Heston is v0 = 0.04, κ = 2, θ = 0.05, ρ = −0.7, vol of vol = 0.5. Strikes in the chain run from 60 to 150. The QuantLib column is that library's engine for the same contract, not the same algorithm: `AnalyticEuropeanEngine`, `AnalyticHestonEngine` (the two single-option Heston rows share one timing; tolerance 1e-12 and 100k evaluations), `FdBlackScholesVanillaEngine`, binomial CRR or Leisen–Reimer, `MCEuropeanEngine`, `MCEuropeanHestonEngine`, and `MCAmericanEngine`. Its Monte Carlo engines step a generic path, which is why those rows are seconds. Leisen–Reimer is timed at 5,001 steps on both sides. Reprint from a clean tree, with QuantLib 1.44 available to the interpreter and not added to the project lock:

```bash
uv run --no-sync --with 'QuantLib==1.44' python benchmarks/run_benchmarks.py --repeats 15 --out benchmarks/results/2026-11-18-arm64.json
```

<!-- perf-table -->
Medians of 15 runs on `macOS-26.6.2-arm64-arm-64bit`, git `ddcc4a9`, Python 3.12.15, NumPy 2.5.4, SciPy 1.18.1, Numba 0.68.1. QuantLib 1.44 times the same contract with its own engine.

| Case | Prices | Median (ms) | QuantLib (ms) |
| --- | --- | ---: | ---: |
| `bsm_closed_form_x1000` | 1,000 European calls, closed form, σ = 0.2 | 1.37 | 1.00 |
| `heston_fft` | one Heston call, Carr–Madan FFT, default grid | 0.35 | 0.16 |
| `heston_integration` | one Heston call, Gil–Pelaez quadrature | 0.06 | 0.16 |
| `cn_pde_200x200` | one European call, Crank–Nicolson, 200 spot × 200 time steps | 0.28 | 0.56 |
| `crr_am_1000` | American put, CRR, 1,000 steps | 9.95 | 1.67 |
| `crr_am_5000` | American put, CRR, 5,000 steps | 160.65 | 38.77 |
| `lr_am_5000` | American put, Leisen–Reimer, 5,001 steps | 161.00 | 122.48 |
| `mc_bsm_100k` | European call, Euler on log S, 100k paths × 100 steps, antithetic | 3.36 | 1678.78 |
| `mc_heston_100k` | Heston call, Andersen QE, 100k paths × 100 steps, antithetic | 23.10 | 2079.49 |
| `lsmc_bsm_20k` | American put, Longstaff–Schwartz degree 2, 20k paths × 50 steps | 11.76 | 205.08 |
| `heston_chain_400` | 400 Heston options, COS, 5 maturities × 40 strikes × call and put | 1.57 | 83.67 |
<!-- /perf-table -->

QuantLib is faster on the CRR and Leisen–Reimer trees, and on one analytic Heston price. The Crank–Nicolson row is the same 200×200 grid. On one core the Black–Scholes Monte Carlo kernel takes 16.59 ms against QuantLib's 1.68 s: the kernel keeps the terminal log-spot, and `MCEuropeanEngine` stores the path. The 3.36 ms in the table is that kernel on 10 cores. The chain is five COS expansions, one per maturity. The 83.67 ms is 400 calls to `AnalyticHestonEngine`, about 0.21 ms each.

---

## Also in the box

* **Models.** Stochastic volatility (Heston, SABR), jumps (Merton, Bates, Kou, SABR with jumps), pure Lévy (VG, NIG, CGMY, Hyperbolic), local volatility (CEV, Dupire), rates (Vasicek, CIR), Black's approximation, the perpetual put, and parity / implied rate.
* **Techniques.** Closed form, COS, Carr–Madan FFT, Gauss–Legendre, Crank–Nicolson, CRR, Leisen–Reimer, trinomial, European Monte Carlo, Longstaff–Schwartz.
* **CLI and dashboard.** `optpricing price` prices one contract from the terminal (`--ticker` downloads a chain and needs a network), and `optpricing calibrate`, `backtest`, and `tune` walk a saved chain without a download; `optpricing dashboard` (the `ui` extra) is the Streamlit app in the gif, also hosted at [optpricing.streamlit.app](https://optpricing.streamlit.app/).

Guides: [Introduction](https://diljit22.github.io/optpricing/guide/introduction/), [Installation](https://diljit22.github.io/optpricing/guide/installation/), [Getting started](https://diljit22.github.io/optpricing/guide/getting_started/), [Methods](docs/guide/methods.md), [Validation](docs/guide/validation.md), [API](https://diljit22.github.io/optpricing/guide/API/), [CLI](https://diljit22.github.io/optpricing/guide/CLI/), [Examples](https://diljit22.github.io/optpricing/guide/examples/).

## Contributing

See [CONTRIBUTING](CONTRIBUTING.md).

## License

MIT. See [LICENSE](LICENSE).
