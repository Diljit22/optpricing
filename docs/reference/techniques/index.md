# Techniques Package

The `techniques` package provides the various numerical and analytical methods used to price options and calculate their sensitivities (Greeks).

Each technique is a concrete implementation of the `BaseTechnique` class and is designed to work with one or more financial models from the `models` package.

-   **Analytical Methods**: `ClosedFormTechnique` for models with exact solutions.
-   **Transform Methods**: `FFTTechnique` and `IntegrationTechnique` for models with a known characteristic function.
-   **Tree-Based Methods**: `CRRTechnique`, `LeisenReimerTechnique`, and `TOPMTechnique` for lattice-based pricing.
-   **Simulation Methods**: `MonteCarloTechnique` for path-based simulation.
-   **Numerical PDE Solvers**: `PDETechnique` for solving the pricing partial differential equation.

For developers, the foundational components are defined in the **[Base Technique](./base/index.md)** section, and select numerical implementations are in the **[Kernels](./kernels/index.md)** section.