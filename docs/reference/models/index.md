# Models Package

The `models` package contains all the financial pricing models available in the library. Each model is a concrete implementation of the `BaseModel` abstract class.

The models can be broadly categorized:

-   **Standard Models**: Foundational models like Black-Scholes-Merton.
-   **Stochastic Volatility**: Models where volatility is its own random process, such as Heston and SABR.
-   **Jump-Diffusion**: Models that incorporate sudden jumps in the asset price, like Merton's Jump-Diffusion and Kou's Double-Exponential model.
-   **Pure Levy**: Models based on Levy processes, such as Variance Gamma (VG), Normal Inverse Gaussian (NIG), and CGMY.

For developers looking to implement new models, please see the **[Base Classes](./base/index.md)** documentation.