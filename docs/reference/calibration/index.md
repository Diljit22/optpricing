# Calibration Package

The `calibration` package provides a comprehensive suite of tools for fitting financial models to market data, estimating parameters from historical data, and calculating implied volatility surfaces.

- **Calibrator**: The main class for orchestrating the optimization process to fit model parameters to market option prices.
- **VolatilitySurface**: A class to compute and manage implied volatility surfaces from both market and model-generated prices.
- **Parameter Fitters**: Utility functions to estimate specific parameters, such as `fit_rate_and_dividend` from put-call parity or `fit_jump_params_from_history` from historical returns.
- **Technique Selector**: A helper function to automatically select the most efficient pricing technique for a given model.
- **IV Solvers**: High-performance, vectorized solvers for calculating implied volatility.
