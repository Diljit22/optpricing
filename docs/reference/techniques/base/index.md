# Base Technique Package

This package contains the foundational abstract classes and helper mixins for all pricing techniques.

-   **BaseTechnique**: The abstract base class that all pricing techniques must inherit from.
-   **LatticeTechnique**: A specialized abstract base class for all tree-based methods, providing common logic for Greek calculations.
-   **GreekMixin**: Provides default numerical implementations for option Greeks (Delta, Gamma, Vega, Theta, Rho).
-   **IVMixin**: Provides a default implementation for calculating implied volatility.
-   **PricingResult**: A simple data container for returning pricing results.
-   **RandomUtils**: Utilities for generating correlated random numbers for Monte Carlo simulations.