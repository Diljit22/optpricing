# Guide: Examples & Benchmarks

The `optpricing` repository includes a set of pre-built benchmark scripts in the `examples/` directory. These scripts are designed to showcase the library's capabilities, compare the performance and accuracy of different numerical techniques, and serve as a starting point for your own custom research.

**Note:** To run these examples, you must have a local clone of the repository. See the [Developer Installation](installation.md#developer-installation) instructions for details.

## Running Benchmarks from the CLI

The easiest way to run these benchmarks is through the `demo` command group in the CLI.

### European Options Benchmark

This benchmark compares closed form, quadrature, and FFT, plus PDE, trees, and Monte Carlo where the model supports them, across the equity models in the script. It does not call `COSTechnique` (the library default for characteristic-function models) and it does not include Dupire. For each model, it generates a detailed table comparing:

- Price
- All primary Greeks (Delta, Gamma, Vega, Theta, Rho)
- Implied Volatility
- Calculation time for each metric

Monte Carlo Greeks are left blank for jump models. The script also prints the put-call parity error for each technique.

**To run the full benchmark suite:**

```bash
optpricing demo european
```

**To run the benchmark for a single model (e.g., Heston):**

```bash
optpricing demo european --model Heston
```

This benchmark is an excellent tool for understanding the trade-offs between speed and accuracy for different numerical methods.

### American Options Benchmark

This benchmark prices American puts on the models Longstaff–Schwartz can simulate (Black–Scholes, Merton, Kou, Heston, Bates, SABR, SABR with jumps). CRR, Leisen–Reimer, and trinomial trees are compared only on Black–Scholes; they reject every other model.

**To run the American options benchmark:**

```bash
optpricing demo american
```

### Interest Rate Models Benchmark

This example demonstrates the use of the library for pricing fixed-income derivatives. It prices a Zero-Coupon Bond using two canonical short-rate models:

- **Vasicek Model**
- **Cox-Ingersoll-Ross (CIR) Model**

It showcases the use of the `ClosedFormTechnique` for models where an analytical solution for bond prices exists.

**To run the interest rate models benchmark:**

```bash
optpricing demo rates
```

These scripts provide a robust demonstration of the library's features and serve as a template for constructing more complex, custom analyses using the `optpricing` API.
