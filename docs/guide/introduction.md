# Introduction

Welcome to the optpricing library! This documentation is your comprehensive guide to understanding and using all the features this library has to offer.

## What is optpricing?

optpricing is a Python library designed to provide a toolkit for quantitative finance, with a primary focus on the pricing and analysis of financial derivatives.

The library is organized around a few core concepts; understanding these will help you navigate the codebase and documentation.

- **Atoms**: These are the fundamental, immutable data structures representing core financial concepts like `Option`, `Stock`, and `Rate`. By using these "atoms," we ensure that data is passed through the system in a consistent and predictable way.

- **Models**: This is a collection of classes representing financial models. It includes everything from the standard Black-Scholes-Merton to advanced models with stochastic volatility (Heston, SABR) and jumps (Merton, Bates, Kou). Each model is a self-contained representation of a specific financial theory.

- **Techniques**: These are the numerical or analytical algorithms used for pricing, such as Monte Carlo simulation, Fast Fourier Transform (FFT), or finite difference methods (PDEs). This design choice decouples the "what" (the model) from the "how" (the pricing algorithm), allowing you to, for example, price a Heston model using either FFT or Monte Carlo.

- **Workflows**: These are high-level orchestrators that combine data, models, and techniques to perform complex, real-world tasks like daily model calibration or historical backtesting. They are the engines that power the command-line interface and the dashboard.

## Our Philosophy

The library was designed with the following principles in mind:

- **Speed**: For computationally intensive tasks like Monte Carlo simulation, we use `numba` to JIT-compile the core numerical kernels, resulting in performance that rivals compiled languages like C or Fortran.

- **Accuracy**: Standard, well-vetted numerical algorithms are used for pricing, calibration, and root-finding, providing a reliable foundation for your analysis.

- **Extensibility**: The object-oriented design, centered around the `BaseModel` and `BaseTechnique` abstract classes, makes it straightforward for you or other developers to add new models or pricing methods without disrupting the existing structure.

- **Usability**: A powerful command-line interface (CLI) and an interactive Streamlit dashboard are provided for common tasks, making the library accessible to users who may not want to write Python code for every analysis.

Ready to get started? Head over to:

- [Getting Started](getting_started.md)
- [API Reference](../reference/atoms/index.md)
- [Interactive Dashboard](dashboard.md)
