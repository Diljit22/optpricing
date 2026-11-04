from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np

from optpricing.techniques.integration import IntegrationTechnique

if TYPE_CHECKING:
    import pandas as pd

    from optpricing.atoms import Rate, Stock
    from optpricing.models import BaseModel

__doc__ = """
Defines a high-performance, vectorized pricer for option chains.

Every model that can be calibrated prices a whole chain through one function,
:func:`price_options_vectorized`, which picks the fastest exact method:

1. a vectorized closed form (BSM, Merton, SABR/Hagan, CEV), one call per maturity;
2. Gil-Pelaez inversion of the characteristic function on a composite
   Gauss-Legendre rule (Heston, Bates, Kou, VG, NIG, CGMY, Hyperbolic), one
   characteristic-function evaluation per maturity;
3. optionally (``allow_mc=True``) Monte Carlo with common random numbers for
   simulation-only models (SABR with jumps, Dupire): one set of terminal spots
   per maturity prices every strike of that maturity.

Repeated pricing of the same chain (calibration) should build a
:class:`ChainLayout` once: it caches the per-maturity strike groups and the
model-independent Fourier kernels ``exp(-i u ln K) / u``.
"""


class ChainLayout:
    """
    Pre-grouped view of an option chain for repeated pricing.

    Parameters
    ----------
    options_df : pd.DataFrame
        Must contain ``strike``, ``maturity`` and ``optionType``; the row order
        defines the output order of the pricers.
    upper_bound : float, optional
        Truncation of the Gil-Pelaez integrals, by default 200.
    n_nodes : int, optional
        Gauss-Legendre nodes per panel, by default 24.
    """

    def __init__(
        self,
        options_df: pd.DataFrame,
        *,
        upper_bound: float = 200.0,
        n_nodes: int = 24,
    ):
        self.n_options = len(options_df)
        maturities = options_df["maturity"].to_numpy(dtype=float)
        strikes = options_df["strike"].to_numpy(dtype=float)
        is_call = options_df["optionType"].to_numpy() == "call"
        self.groups: list[tuple[float, np.ndarray, np.ndarray, np.ndarray]] = []
        if self.n_options:
            # Stable first-appearance order of maturities (like groupby(sort=False)).
            _, first_idx, inverse = np.unique(
                maturities, return_index=True, return_inverse=True
            )
            for gid in np.argsort(first_idx):
                loc = np.flatnonzero(inverse == gid)
                self.groups.append(
                    (float(maturities[loc[0]]), loc, strikes[loc], is_call[loc])
                )
        self.u, self.w = IntegrationTechnique.gauss_nodes(upper_bound, n_nodes)
        self._twist: list[np.ndarray | None] = [None] * len(self.groups)

    def twist(self, i: int) -> np.ndarray:
        """``exp(-i u ln K) / u`` for maturity group ``i`` (cached)."""
        cached = self._twist[i]
        if cached is None:
            lnK = np.log(self.groups[i][2])
            cached = np.exp(-1j * np.outer(lnK, self.u)) / self.u
            self._twist[i] = cached
        return cached


def _closed_form_chain(
    layout: ChainLayout, stock: Stock, model: BaseModel, rate: Rate, prices: np.ndarray
) -> None:
    S, q = stock.spot, stock.dividend
    for T, loc, K, is_call in layout.groups:
        prices[loc] = model.price_closed_form(
            spot=S, strike=K, r=rate.get_rate(T), q=q, t=T, call=is_call
        )


def _characteristic_function_chain(
    layout: ChainLayout,
    stock: Stock,
    model: BaseModel,
    rate: Rate,
    prices: np.ndarray,
    **kwargs: Any,
) -> None:
    S, q = stock.spot, stock.dividend
    u, w = layout.u, layout.w
    n = len(u)
    # One characteristic-function call per maturity: nodes for P2, shifted
    # nodes for P1 and the martingale normaliser phi(-i).
    u_all = np.concatenate([u, u - 1j, [-1j]])
    for i, (T, loc, K, is_call) in enumerate(layout.groups):
        r = rate.get_rate(T)
        with np.errstate(all="ignore"):
            # Far-from-valid parameter sets can overflow or hit a removable
            # singularity (e.g. Heston at u=-i when rho*xi > kappa); the
            # resulting NaNs are handled by the caller (calibration penalty).
            phi_all = model.cf(t=T, spot=S, r=r, q=q, **kwargs)(u_all)
        twist = layout.twist(i)  # (n_strikes, n_nodes)

        # Gil-Pelaez probabilities on the composite Gauss-Legendre rule.
        p2 = (twist * phi_all[:n]).imag @ w
        p1 = (twist * phi_all[n : 2 * n]).imag @ w

        # phi(-i) = E[S_T]; for a risk-neutral model this is the forward, which
        # is also the fallback when the characteristic function is degenerate there.
        denom = np.real(phi_all[-1])
        if not np.isfinite(denom) or abs(denom) < 1e-12:
            denom = S * math.exp((r - q) * T)

        P1 = 0.5 + p1 / (np.pi * denom)
        P2 = 0.5 + p2 / np.pi

        df_q, df_r = math.exp(-q * T), math.exp(-r * T)
        call_vals = S * df_q * P1 - K * df_r * P2
        put_vals = K * df_r * (1.0 - P2) - S * df_q * (1.0 - P1)
        prices[loc] = np.where(is_call, call_vals, put_vals)


def _monte_carlo_terminal_spots(
    model: BaseModel,
    S0: float,
    r: float,
    q: float,
    T: float,
    *,
    n_paths: int,
    n_steps: int,
    seed: int,
    **kwargs: Any,
) -> np.ndarray:
    """Terminal spots with the same dispatch as `MonteCarloTechnique.price`."""
    from optpricing.techniques.monte_carlo import MonteCarloTechnique

    mc = MonteCarloTechnique(n_paths=n_paths, n_steps=n_steps, seed=seed)
    if getattr(model, "has_exact_sampler", False):
        return model.sample_terminal_spot(S0, r, T, mc.n_paths, q=q, rng=mc.rng)
    if getattr(model, "is_pure_levy", False):
        return mc._simulate_levy_terminal(model, S0, r, q, T)
    return mc._simulate_sde_path(model, S0, r, q, T, **kwargs)


def _monte_carlo_chain(
    layout: ChainLayout,
    stock: Stock,
    model: BaseModel,
    rate: Rate,
    prices: np.ndarray,
    *,
    n_paths: int,
    n_steps: int,
    seed: int,
    **kwargs: Any,
) -> None:
    S, q = stock.spot, stock.dividend
    for i, (T, loc, K, is_call) in enumerate(layout.groups):
        r = rate.get_rate(T)
        # One stream per maturity, fixed by the seed: common random numbers
        # across optimiser iterations keep the objective smooth.
        ST = _monte_carlo_terminal_spots(
            model, S, r, q, T, n_paths=n_paths, n_steps=n_steps, seed=seed + i, **kwargs
        )
        payoff_call = np.maximum(ST[None, :] - K[:, None], 0.0).mean(axis=1)
        payoff_put = np.maximum(K[:, None] - ST[None, :], 0.0).mean(axis=1)
        prices[loc] = math.exp(-r * T) * np.where(is_call, payoff_call, payoff_put)


def supports_vectorized_pricing(model: BaseModel, *, allow_mc: bool = False) -> bool:
    """
    True if :func:`price_options_vectorized` can price a chain for ``model``.

    Parameters
    ----------
    model : BaseModel
        The model to check.
    allow_mc : bool, optional
        Count simulation-only models as supported, by default False.
    """
    if model.has_closed_form and getattr(model, "vectorized_closed_form", False):
        return True
    if model.supports_cf:
        return True
    return allow_mc and (
        model.supports_sde
        or getattr(model, "is_pure_levy", False)
        or getattr(model, "has_exact_sampler", False)
    )


def price_chain(
    layout: ChainLayout,
    stock: Stock,
    model: BaseModel,
    rate: Rate,
    *,
    allow_mc: bool = False,
    mc_paths: int = 20_000,
    mc_steps: int = 50,
    mc_seed: int = 0,
    **kwargs: Any,
) -> np.ndarray:
    """
    Prices a pre-grouped chain (see :class:`ChainLayout`).

    Parameters are as for :func:`price_options_vectorized`.
    """
    prices = np.empty(layout.n_options)
    if layout.n_options == 0:
        return prices
    if model.has_closed_form and getattr(model, "vectorized_closed_form", False):
        _closed_form_chain(layout, stock, model, rate, prices)
    elif model.supports_cf:
        _characteristic_function_chain(layout, stock, model, rate, prices, **kwargs)
    elif allow_mc and supports_vectorized_pricing(model, allow_mc=True):
        _monte_carlo_chain(
            layout,
            stock,
            model,
            rate,
            prices,
            n_paths=mc_paths,
            n_steps=mc_steps,
            seed=mc_seed,
            **kwargs,
        )
    else:
        raise NotImplementedError(
            f"{model.name} has no closed form or characteristic function; pass "
            "allow_mc=True to price it by Monte Carlo."
        )
    return prices


def price_options_vectorized(
    options_df: pd.DataFrame,
    stock: Stock,
    model: BaseModel,
    rate: Rate,
    *,
    upper_bound: float = 200.0,
    allow_mc: bool = False,
    mc_paths: int = 20_000,
    mc_steps: int = 50,
    mc_seed: int = 0,
    **kwargs: Any,
) -> np.ndarray:
    """
    Prices a chain of European options with the fastest exact method available.

    Parameters
    ----------
    options_df : pd.DataFrame
        Must contain `strike`, `maturity` and `optionType`.
        The index is reset internally to guarantee safe positional writes.
    stock : Stock
        Underlying description.
    model : BaseModel
        A model with a vectorized closed form, a characteristic function or
        (with ``allow_mc``) a simulation method.
    rate : Rate
        Continuous zero-curve.
    upper_bound : float, default 200
        Integration truncation limit for the characteristic-function route.
    allow_mc : bool, default False
        Fall back to Monte Carlo for simulation-only models.
    mc_paths, mc_steps, mc_seed : int
        Monte Carlo settings for that fallback (fixed seed: common random
        numbers across calls).
    **kwargs : Any
        Extra pricing arguments forwarded to the characteristic function or
        simulation kernel (e.g. ``v0``).

    Returns
    -------
    np.ndarray
        Model prices - aligned with the row order of options_df.

    Raises
    ------
    NotImplementedError
        If the model has neither a vectorized closed form nor a characteristic
        function and ``allow_mc`` is False.
    """
    layout = ChainLayout(options_df, upper_bound=upper_bound)
    return price_chain(
        layout,
        stock,
        model,
        rate,
        allow_mc=allow_mc,
        mc_paths=mc_paths,
        mc_steps=mc_steps,
        mc_seed=mc_seed,
        **kwargs,
    )
