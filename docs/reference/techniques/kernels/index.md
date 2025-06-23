# Kernels Package

This package contains the low-level, high-performance numerical implementations that power the pricing techniques. These functions are designed to be pure and operate on primitive data types, making them ideal for JIT-compilation with `numba`.

-   **Lattice Kernels**: The core tree-building and backward-induction logic for binomial and trinomial models.
-   **MC Kernels**: JIT-compiled functions for simulating the stochastic differential equations (SDEs) of various models.