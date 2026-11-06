from __future__ import annotations

from typing import Any

import numpy as np

from optpricing.models.base import BaseModel, ParamValidator

__doc__ = """
Defines the Constant Elasticity of Variance (CEV) model.
"""

_LOGNORMAL_WINDOW = 1e-3


class CEVModel(BaseModel):
    """
    Constant Elasticity of Variance (CEV) model, ``dS = (r-q) S dt + sigma S^gamma dW``.

    European options have the Cox (1975) / Schroder (1989) closed form in terms
    of the noncentral chi-squared distribution (``price_closed_form``); the
    terminal spot can be sampled exactly (``sample_terminal_spot``). For
    ``gamma < 1`` the origin is absorbing, which both methods honour.
    """

    name: str = "CEV"
    supports_sde: bool = True
    has_exact_sampler: bool = True
    has_closed_form: bool = True
    vectorized_closed_form: bool = True
    default_params = {"sigma": 0.8, "gamma": 0.7}
    param_defs = {
        "sigma": {
            "label": "Volatility scale",
            "default": 0.8,
            "min": 0.01,
            "max": 5.0,
            "step": 0.01,
        },
        "gamma": {
            "label": "Elasticity",
            "default": 0.7,
            "min": 0.0,
            "max": 2.0,
            "step": 0.05,
        },
    }

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

    def _closed_form_impl(
        self,
        *,
        spot: float,
        strike: float | np.ndarray,
        r: float,
        q: float,
        t: float,
        call: bool | np.ndarray = True,
        **_: Any,
    ) -> float | np.ndarray:
        """
        Cox/Schroder closed form for European options under CEV.

        With ``a = (K e^{-(r-q)T})^{2(1-g)} / ((1-g)^2 v)``, ``b = 1/(1-g)``,
        ``c = S^{2(1-g)} / ((1-g)^2 v)`` and
        ``v = sigma^2 (e^{2(r-q)(g-1)T} - 1) / (2(r-q)(g-1))`` (``sigma^2 T`` when
        ``r = q``), the call is (Hull, *Options, Futures and Other Derivatives*,
        CEV section; Schroder 1989)

        * ``g < 1``: ``S e^{-qT} [1 - X(a; b+2, c)] - K e^{-rT} X(c; b, a)``
        * ``g > 1``: ``S e^{-qT} [1 - X(c; -b, a)] - K e^{-rT} X(a; 2-b, c)``

        where ``X(z; k, nc)`` is the noncentral chi-squared CDF with ``k`` degrees
        of freedom and non-centrality ``nc``. For ``g > 1`` the spot is a strict
        local martingale, ``E[S_T] = S e^{(r-q)T} P(1/(2(g-1)), c/2)`` (``P`` the
        regularised lower incomplete gamma; inverse-Bessel identity), and the
        formula above exceeds the expectation ``E[e^{-rT}(S_T-K)^+]`` by
        ``S e^{-qT} (1 - P)``; prices are the expectations (as QuantLib's
        ``AnalyticCEVEngine`` and the exact sampler), so put-call parity reads
        ``C - P = e^{-rT} E[S_T] - K e^{-rT}``. For ``g`` within 1e-3 of 1, where
        the noncentral chi-squared CDFs lose precision, the price is interpolated
        quadratically in ``g`` between the formula at ``1 -/+ 1e-3`` and
        Black-Scholes at ``g = 1`` (within 1e-7 of QuantLib's ``AnalyticCEVEngine``).

        Parameters
        ----------
        spot : float
            The current price of the underlying asset.
        strike : float | np.ndarray
            The strike price(s).
        r : float
            The continuously compounded risk-free rate.
        q : float
            The continuously compounded dividend yield.
        t : float
            The time to maturity of the option, in years.
        call : bool | np.ndarray, optional
            True for a call option, False for a put; a boolean array prices a
            mix of calls and puts for array ``strike``. Defaults to True.

        Returns
        -------
        float | np.ndarray
            The price(s) of the European option(s).
        """
        from scipy.special import gammainc, ndtr
        from scipy.stats import ncx2

        p = self.params
        sigma, gamma = p["sigma"], p["gamma"]
        scalar = np.ndim(strike) == 0 and not isinstance(call, np.ndarray)
        K = np.asarray(strike, dtype=float)
        mu = r - q
        df_div, df_rate = np.exp(-q * t), np.exp(-r * t)

        def lognormal_call() -> np.ndarray:
            sqrt_t = np.sqrt(t)
            d1 = (np.log(spot / K) + (mu + 0.5 * sigma**2) * t) / (sigma * sqrt_t)
            d2 = d1 - sigma * sqrt_t
            return spot * df_div * ndtr(d1) - K * df_rate * ndtr(d2)

        def cev_call(g: float) -> tuple[np.ndarray, float]:
            """Call expectation and the discounted expected terminal spot."""
            one_minus = 1.0 - g
            exponent = 2.0 * mu * (g - 1.0) * t
            v = (
                sigma**2 * t
                if abs(exponent) < 1e-12
                else sigma**2 * np.expm1(exponent) / (exponent / t)
            )
            scale = one_minus**2 * v
            a = (K * np.exp(-mu * t)) ** (2.0 * one_minus) / scale
            b = 1.0 / one_minus
            c = spot ** (2.0 * one_minus) / scale
            if g < 1.0:
                call_px = spot * df_div * (1.0 - ncx2.cdf(a, b + 2.0, c)) - (
                    K * df_rate * ncx2.cdf(c, b, a)
                )
                return call_px, spot * df_div
            pv_spot = spot * df_div * gammainc(0.5 / (g - 1.0), 0.5 * c)
            call_px = (
                pv_spot
                - spot * df_div * ncx2.cdf(c, -b, a)
                - (K * df_rate * ncx2.cdf(a, 2.0 - b, c))
            )
            return call_px, pv_spot

        h = _LOGNORMAL_WINDOW
        x = gamma - 1.0
        if x == 0.0:
            call_px, pv_spot = lognormal_call(), spot * df_div
        elif abs(x) < h:
            # ncx2 loses all precision as gamma -> 1 (NaN below ~1e-5): quadratic
            # interpolation between the exact formula at 1 -/+ h and Black-Scholes
            # at gamma = 1, error O(h^3).
            (lo, pv_lo), (hi, pv_hi) = cev_call(1.0 - h), cev_call(1.0 + h)
            w_lo, w_hi = x * (x - h) / (2.0 * h * h), x * (x + h) / (2.0 * h * h)
            w_mid = 1.0 - w_lo - w_hi
            call_px = w_lo * lo + w_mid * lognormal_call() + w_hi * hi
            pv_spot = w_lo * pv_lo + w_mid * spot * df_div + w_hi * pv_hi
        else:
            call_px, pv_spot = cev_call(gamma)
        put_px = call_px - pv_spot + K * df_rate

        if isinstance(call, np.ndarray):
            price = np.where(call, call_px, put_px)
        else:
            price = call_px if call else put_px
        return float(price) if scalar else price

    #  Abstract Method Implementations
    def _sde_impl(self, **kwargs: Any) -> Any:
        raise NotImplementedError("CEV uses a specialized exact sampler.")

    def _cf_impl(self, **kwargs: Any) -> Any:
        raise NotImplementedError

    def _pde_impl(self, **kwargs: Any) -> Any:
        raise NotImplementedError
