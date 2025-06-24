# Workflows Package

The `workflows` package provides high-level classes that orchestrate complex, multi-step processes. These are the primary engines for running calibrations and backtests.

The workflows are designed to be driven by a user interface, such as the command-line interface or the Streamlit dashboard.

-   **DailyWorkflow**: Encapsulates the logic for calibrating a model on a single day's market data.
-   **BacktestWorkflow**: Manages the process of running a `DailyWorkflow` over a series of historical dates to evaluate a model's out-of-sample performance.

This package also contains a `configs` sub-package, which holds the specific "recipes" (initial parameters, bounds, etc.) for calibrating each supported financial model.