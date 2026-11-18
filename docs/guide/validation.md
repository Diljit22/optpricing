# Validation

The prices below are the contracts in `tests/correctness/`, repriced by
`benchmarks/accuracy.py`. QuantLib is not a dependency. Where a test pins a
QuantLib 1.44 value, that number is the reference. The other references are a
closed form or an independent COS evaluation, and the test file names which.

```bash
uv run python benchmarks/accuracy.py
```

That command reprints every number in the table. `--write` refreshes
`benchmarks/accuracy.json` and the copies in this page and the README.
`--check` reprices and fails if a committed error moves by more than 5%, or if
the markdown no longer matches the JSON.

<!-- accuracy-summary-full -->
| Model | Technique | Reference | Cases | Max abs error | Worst case |
| --- | --- | --- | ---: | ---: | --- |
| Heston | COS | QuantLib 1.44 AnalyticHestonEngine | 18 | 5.0e-10 | atm_put |
| Heston | Gauss-Legendre | QuantLib 1.44 AnalyticHestonEngine | 18 | 5.0e-10 | atm_put |
| Heston | FFT | QuantLib 1.44 AnalyticHestonEngine | 5 | 5.0e-10 | atm_put |
| Heston | chain COS | QuantLib 1.44 AnalyticHestonEngine | 4 | 7.0e-10 | T=0.0273973 K=110 |
| Heston | chain Gauss-Legendre | QuantLib 1.44 AnalyticHestonEngine | 4 | 4.1e-11 | T=0.10137 K=110 |
| NIG | COS | COS, N=2^18 (Fang and Oosterlee truncation) | 3 | 8.2e-11 | T=0.1 K=90 |
| NIG | Gauss-Legendre | COS, N=2^18 (Fang and Oosterlee truncation) | 3 | 8.2e-11 | T=0.1 K=90 |
| NIG | chain COS | COS, N=2^18 (Fang and Oosterlee truncation) | 3 | 5.2e-10 | T=0.1 K=100 |
| NIG | chain Gauss-Legendre | COS, N=2^18 (Fang and Oosterlee truncation) | 3 | 8.2e-11 | T=0.1 K=90 |
| Hyperbolic | COS | COS, N=2^18 (Fang and Oosterlee truncation) | 1 | 1.6e-11 | T=0.1 K=100 |
| Hyperbolic | Gauss-Legendre | COS, N=2^18 (Fang and Oosterlee truncation) | 1 | 1.6e-11 | T=0.1 K=100 |
| Hyperbolic | chain COS | COS, N=2^18 (Fang and Oosterlee truncation) | 1 | 3.5e-11 | T=0.1 K=100 |
| Hyperbolic | chain Gauss-Legendre | COS, N=2^18 (Fang and Oosterlee truncation) | 1 | 1.6e-11 | T=0.1 K=100 |
| Kou | COS | COS, N=2^18, on the paper characteristic function | 3 | 2.3e-11 | K=80 put |
| Kou | Gauss-Legendre | COS, N=2^18, on the paper characteristic function | 3 | 2.3e-11 | K=80 put |
| NIG | COS | COS, N=2^18, on the paper characteristic function | 3 | 4.5e-11 | K=100 call |
| NIG | Gauss-Legendre | COS, N=2^18, on the paper characteristic function | 3 | 4.5e-11 | K=100 call |
| Hyperbolic | COS | COS, N=2^18, on the paper characteristic function | 3 | 4.3e-11 | K=100 call |
| Hyperbolic | Gauss-Legendre | COS, N=2^18, on the paper characteristic function | 3 | 4.3e-11 | K=100 call |
| VG | COS | Fang and Oosterlee (2008) | 1 | 2.0e-10 | VG |
| VG | Gauss-Legendre | Fang and Oosterlee (2008) | 1 | 2.0e-10 | VG |
| CGMY | COS | Fang and Oosterlee (2008) | 2 | 4.8e-10 | CGMY Y=1.5 |
| CGMY | Gauss-Legendre | Fang and Oosterlee (2008) | 2 | 4.8e-10 | CGMY Y=1.5 |
| Kou | FFT | COS, N=2^18, on the paper characteristic function | 6 | 2.7e-11 | kou_eta1_2.5 K=100 |
| CGMY | FFT | COS, N=2^18, on the paper characteristic function | 4 | 5.3e-10 | cgmy_fo_y1.98 K=100 |
| NIG | FFT | COS, N=2^18, on the paper characteristic function | 3 | 4.5e-11 | nig_skewed K=80 |
| VG | COS | COS, N=2^22, fixed range | 3 | 1.1e-07 | T=0.05 K=100 |
| CGMY | COS | COS, N=2^22, fixed range | 3 | 5.3e-07 | T=0.05 K=100 |
| Bates | COS | QuantLib 1.44 BatesEngine | 6 | 4.4e-11 | kappa=0.3 K=100 |
| Bates | Gauss-Legendre | QuantLib 1.44 BatesEngine | 8 | 3.6e-10 | otm_put |
| Merton | closed form | Merton (1976) series, 80 terms | 2 | 4.6e-10 | otm_put |
| Merton | Gauss-Legendre | Merton (1976) series, 80 terms | 2 | 4.8e-10 | otm_put |
| Merton | closed form | Gil-Pelaez quadrature | 1 | 1.0e-11 | lambda T = 240 |
| Vasicek | closed form | QuantLib 1.44 discount bond | 4 | 4.9e-11 | vasicek_10y face=100 |
| CIR | closed form | QuantLib 1.44 discount bond | 4 | 4.1e-11 | cir_10y face=100 |
| Black's approximation | closed form | Hull, Black's approximation | 4 | 4.8e-11 | kwargs order=-1 |
| Perpetual put | closed form | Merton (1973) perpetual put | 8 | 7.1e-15 | sigma=0.4 S=120 |
| CEV | closed form | Hull CEV formula (zero rates also QuantLib 1.44) | 3 | 4.2e-10 | gamma_0.5_zero_rates |
| CEV | Monte Carlo | Hull CEV formula (zero rates also QuantLib 1.44) | 3 | 8.6e-03 | gamma_0.8_with_dividend |
| CEV | closed form | QuantLib 1.44 AnalyticCEVEngine | 27 | 6.0e-08 | gamma=0.9995 K=120 |
| BSM | CRR | BSM closed form | 1 | 2.0e-03 | S=K=100 |
| BSM | Leisen-Reimer | BSM closed form | 1 | 1.5e-06 | S=K=100 |
| BSM | TOPM | BSM closed form | 1 | 9.8e-04 | S=K=100 |
| BSM | Crank-Nicolson | BSM closed form | 1 | 5.4e-03 | S=K=100 |
| BSM American | CRR | QuantLib 1.44 FdBlackScholesVanillaEngine | 2 | 4.8e-04 | atm |
| BSM American | Leisen-Reimer | QuantLib 1.44 FdBlackScholesVanillaEngine | 2 | 2.9e-04 | atm |
| BSM American | TOPM | QuantLib 1.44 FdBlackScholesVanillaEngine | 2 | 5.6e-04 | itm_with_dividend |
| BSM American | LSMC | QuantLib 1.44 FdBlackScholesVanillaEngine | 2 | 2.1e-02 | atm |
| BSM | FFT | COSTechnique | 3 | 1.8e-13 | K=80 T=0.25 |
| BSM | Gauss-Legendre | COSTechnique | 3 | 4.3e-14 | K=120 T=2 |
| BSM | COS | BSM closed form | 3 | 6.4e-14 | K=120 T=2 |
| Heston | FFT | COSTechnique | 3 | 7.8e-14 | K=80 T=0.25 |
| Heston | Gauss-Legendre | COSTechnique | 3 | 4.3e-14 | K=100 T=1 |
| Merton | FFT | COSTechnique | 3 | 1.6e-13 | K=80 T=0.25 |
| Merton | Gauss-Legendre | COSTechnique | 3 | 6.4e-14 | K=100 T=1 |
| Bates | FFT | COSTechnique | 3 | 1.2e-13 | K=100 T=1 |
| Bates | Gauss-Legendre | COSTechnique | 3 | 4.3e-14 | K=120 T=2 |
| Kou | FFT | COSTechnique | 3 | 1.2e-10 | K=80 T=0.25 |
| Kou | Gauss-Legendre | COSTechnique | 3 | 1.4e-14 | K=120 T=2 |
| VG | FFT | COSTechnique | 3 | 1.5e-13 | K=120 T=2 |
| VG | Gauss-Legendre | COSTechnique | 3 | 6.2e-08 | K=80 T=0.25 |
| NIG | FFT | COSTechnique | 3 | 1.0e-13 | K=100 T=1 |
| NIG | Gauss-Legendre | COSTechnique | 3 | 7.1e-14 | K=100 T=1 |
| CGMY | FFT | COSTechnique | 3 | 1.7e-13 | K=100 T=1 |
| CGMY | Gauss-Legendre | COSTechnique | 3 | 4.3e-14 | K=100 T=1 |
| Hyperbolic | FFT | COSTechnique | 3 | 1.8e-13 | K=80 T=0.25 |
| Hyperbolic | Gauss-Legendre | COSTechnique | 3 | 6.4e-14 | K=120 T=2 |
<!-- /accuracy-summary-full -->

Heston against `AnalyticHestonEngine` is 5e-10 (7e-10 on the vectorised chain).
The largest closed-form gap on a QuantLib number is CEV near γ = 1, at 6e-8;
that test allows 1e-7. American trees are measured against QuantLib's
finite-difference engine and sit around 1e-4 to 1e-3. Longstaff–Schwartz and
the CEV exact sampler are sampling error, not a discretisation bug. The block
of FFT and Gauss–Legendre rows at the bottom is technique against technique,
measured on COS for the same contract. Heston FFT and COS agree to about
1e-13 on that grid; Kou's FFT gap on it is 1e-10.

## Convergence

Slopes are \(\Delta \log|\text{error}| / \Delta \log N\) against the
Black–Scholes closed form, on the grids in `test_convergence.py`. CRR and TOPM
are first order at the money. Leisen–Reimer (odd step counts) and
Crank–Nicolson with the Rannacher start are second order. The Monte Carlo row
is the slope of the seed-to-seed standard deviation against the number of
paths, which is \(-1/2\) for independent draws. One step, no antithetics, 60
seeds, the same seeds as the test.

<!-- accuracy-convergence -->
| Scheme | Resolutions | Measured slopes | Mean | Theory |
| --- | --- | --- | ---: | ---: |
| CRR | 200, 400, 800 | -1.00, -1.00 | -1.00 | -1.00 |
| TOPM | 200, 400, 800 | -1.00, -1.00 | -1.00 | -1.00 |
| Leisen-Reimer | 101, 201, 401 | -1.99, -1.99 | -1.99 | -2.00 |
| Crank-Nicolson | 100, 200, 400 | -2.05, -1.98 | -2.01 | -2.00 |
| Monte Carlo | 2,048 and 32,768 paths | -0.45 | -0.45 | -0.50 |
<!-- /accuracy-convergence -->
