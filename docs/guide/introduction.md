# Introduction

Welcome to `optpricing`, a Python toolkit for pricing and calibrating financial derivatives. This library was created to implement and understand the mathematical and computational foundations of quantitative finance, growing into a robust, extensible framework.

## Guiding Principles

The library is organized around four core concepts; understanding these will help you navigate the codebase and documentation.

1. **Atoms**: Immutable data structures for core financial concepts like `Option`, `Stock`, and `Rate`. They provide a consistent foundation for every calculation. This ensures clarity of inputs across the entire library.

2. **Models**: An extensible module of financial models, including classical option pricing models, advanced stochastic volatility models, jump-diffusion processes, and interest rate models. In addition to pricing options, some models support valuation of implied rates, volatility-focused analysis, and put-call parity. Each model is a self-contained representation of a specific financial theory.

3. **Techniques**: These are the numerical algorithms used for pricing models, with bespoke Greek calculations or fallback to numerical differentiation. The separation of model (the "what") from technique (the "how") is a core design feature. The library includes a wide arrange of techniques.

4. **Workflows & Tooling**: High-level orchestrators that combine data, models, and techniques to perform complex, real-world tasks like daily model calibration or historical backtesting. These power the command-line interface and the Streamlit dashboard.

## Who Is This For?

This library is designed for anyone interested in the intersection of finance, mathematics, and software engineering. In particular, the object-oriented design, centered around the `BaseModel` and `BaseTechnique` abstract classes, makes it straightforward to add new models or pricing methods. Benchmarks facilitate performance comparisons with existing models.

---

If you are ready to get started head to the [Installation guide](installation.md).
