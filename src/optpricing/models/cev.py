from __future__ import annotations

from typing import Any

import numpy as np

from optpricing.models.base import BaseModel, ParamValidator

__doc__ = """
Defines the Constant Elasticity of Variance (CEV) model.
"""


class CEVModel(BaseModel):
    """Constant Elasticity of Variance (CEV) model."""

    name: str = "CEV"
    supports_sde: bool = True
    has_exact_sampler: bool = True
    default_params = {"sigma": 0.8, "gamma": 0.7}

    def __init__(self, params: dict[str, float] | None = None):
        """
        Initializes the CEV model.

        Parameters
        ----------
        params : dict[str, float] | None, optional
            A dictionary of model parameters. If None, `default_params` are used.
            Must contain 'sigma' and 'gamma'. Defaults to None.
        """
        super().__init__(params or self.default_params)

    def _validate_params(self) -> None:
        """Validates that 'sigma' and 'gamma' are present and positive."""
        p = self.params
        req = ["sigma", "gamma"]
        ParamValidator.require(p, req, model=self.name)
        ParamValidator.positive(p, ["sigma"], model=self.name)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CEVModel):
            return NotImplemented
        return self.params == other.params

    def __hash__(self) -> int:
        return hash((self.__class__, tuple(sorted(self.params.items()))))

    def sample_terminal_spot(
        self,
        S0: float,
        r: float,
        T: float,
        size: int,
        q: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """
        Exact simulation of the terminal spot of ``dS = (r-q)S dt + sigma S^gamma dW``.

        The drifted CEV process is a deterministic time change of the driftless
        one, ``S_T = exp((r-q)T) F_tau`` with
        ``tau = (exp(2(r-q)(gamma-1)T) - 1) / (2(r-q)(gamma-1))``, and
        ``X = F^(2(1-gamma)) / ((1-gamma)^2 sigma^2)`` is a squared Bessel
        process of dimension ``delta = (1-2 gamma)/(1-gamma)``.

        * ``gamma < 1`` (``delta < 2``): the origin is attainable and absorbing;
          ``X_tau`` is drawn from the absorbed BESQ law (Makarov & Glew 2010):
          with ``lam = X_0 / (2 tau)`` and ``a = 1 - delta/2``, draw
          ``G ~ Gamma(a, 1)``; the path is absorbed iff ``G >= lam`` (probability
          ``Q(a, lam)``), otherwise ``N ~ Poisson(lam - G)`` and
          ``X_tau ~ Gamma(N + 1, 2 tau)``. This reproduces Cox's (1975) /
          Schroder's (1989) pricing formula and keeps ``S`` a true martingale.
        * ``gamma > 1`` (``delta > 2``): the origin is unattainable and
          ``X_tau / tau`` is noncentral chi-squared with ``delta`` degrees of
          freedom and non-centrality ``X_0 / tau``.
        * ``gamma == 1`` reduces to geometric Brownian motion.

        Parameters
        ----------
        S0 : float
            The initial spot price of the asset.
        r : float
            The continuously compounded risk-free rate.
        T : float
            The time to maturity, in years.
        size : int
            The number of samples to generate.
        q : float, optional
            The continuously compounded dividend yield, by default 0.0.
        rng : np.random.Generator | None, optional
            Random number generator; a fresh default generator is used if None.

        Returns
        -------
        np.ndarray
            An array of simulated terminal spot prices (zeros for absorbed paths).
        """
        from scipy.stats import ncx2

        rng = np.random.default_rng() if rng is None else rng
        p = self.params
        sigma, gamma = p["sigma"], p["gamma"]
        mu = r - q

        # gamma -> 1 is geometric Brownian motion
        if abs(1.0 - gamma) < 1e-6:
            drift = (mu - 0.5 * sigma**2) * T
            diffusion = sigma * np.sqrt(T) * rng.standard_normal(size)
            return S0 * np.exp(drift + diffusion)

        exponent = 2.0 * mu * (gamma - 1.0) * T
        tau = T if abs(exponent) < 1e-12 else np.expm1(exponent) / (exponent / T)

        n = 2.0 * (1.0 - gamma)
        scale = (1.0 - gamma) ** 2 * sigma**2
        x0 = S0**n / scale
        delta = (1.0 - 2.0 * gamma) / (1.0 - gamma)

        if gamma < 1.0:
            lam = x0 / (2.0 * tau)
            a = 1.0 - 0.5 * delta
            g = rng.gamma(shape=a, scale=1.0, size=size)
            alive = g < lam
            n_jumps = rng.poisson(np.maximum(lam - g, 0.0))
            x = np.zeros(size)
            x[alive] = rng.gamma(shape=n_jumps[alive] + 1.0, scale=2.0 * tau)
        else:
            x = tau * ncx2.rvs(df=delta, nc=x0 / tau, size=size, random_state=rng)

        f_tau = (scale * x) ** (1.0 / n)
        return np.exp(mu * T) * f_tau

    #  Abstract Method Implementations
    def _sde_impl(self, **kwargs: Any) -> Any:
        raise NotImplementedError("CEV uses a specialized exact sampler.")

    def _cf_impl(self, **kwargs: Any) -> Any:
        raise NotImplementedError

    def _pde_impl(self, **kwargs: Any) -> Any:
        raise NotImplementedError

    def _closed_form_impl(self, **kwargs: Any) -> Any:
        raise NotImplementedError
