# Guide: Examples & Benchmarks

The `optpricing` repository includes a set of pre-built benchmark scripts in the `examples/` directory. These scripts are designed to showcase the library's capabilities, compare the performance and accuracy of different numerical techniques, and serve as a starting point for your own custom research.

**Note:** To run these examples, you must have a local clone of the repository. See the [Developer Installation](installation.md#developer-installation) instructions for details.

## Running Benchmarks from the CLI

The easiest way to run these benchmarks is through the `demo` command group in the CLI.

### European Options Benchmark

This is the most comprehensive benchmark, designed to compare a wide array of pricing techniques across the full suite of European option models. For each model, it generates a detailed table comparing:

- Price
- All primary Greeks (Delta, Gamma, Vega, Theta, Rho)
- Implied Volatility
- Calculation time for each metric

It also calculates the put-call parity error for each technique, serving as a powerful validation of the implementation's correctness.

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

This benchmark focuses specifically on techniques suitable for pricing American-style options, which allow for early exercise. It compares the results of:

- **Longstaff-Schwartz Monte Carlo**: A flexible, simulation-based approach.
- **Lattice/Tree Methods**: Including CRR, Leisen-Reimer, and Trinomial trees.

The script prices a standard American put option across all supported models and presents a comparison of the calculated price and performance.

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
