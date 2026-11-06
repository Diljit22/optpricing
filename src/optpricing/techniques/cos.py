from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np

from optpricing.atoms import Option, OptionType, Rate, Stock
from optpricing.models import BaseModel
from optpricing.techniques.base import BaseTechnique, GreekMixin, IVMixin, PricingResult

__doc__ = """
Defines the COS method of Fang & Oosterlee (2008): the density of ``ln S_T`` is
expanded in a Fourier-cosine series on a truncation range, whose coefficients
are read off the characteristic function, and the put payoff is integrated
against it in closed form.
"""

_CUMULANT_STEPS = 2.0 ** np.arange(-14, 5)
_CHERNOFF_ORDERS = 2.0 ** np.arange(-3.0, 12.0, 0.5)
_MIN_TERMS = 64
_TAIL_WARN = 1e-6


def _cumulant_range(phi: Any, log_forward: float, width: float) -> tuple[float, float]:
    """
    ``c1 -/+ width sqrt(c2 + sqrt(c4))`` (Fang & Oosterlee 2008, eq. 49), with
    the cumulants of ``ln S_T`` from two-step differences of ``psi(u) = log
    phi(u) - i u log_forward`` at the largest grid step ``h`` with ``|psi(2h)|
    <= 1/2``: ``Re psi(h) = -c2 h^2/2 + c4 h^4/24``, ``Im psi(h) = (c1 -
    log_forward) h - c3 h^3/6``.
    """
    h = _CUMULANT_STEPS
    with np.errstate(all="ignore"):
        values = np.asarray(phi(np.concatenate([h, 2.0 * h])), dtype=complex)
        psi = np.log(values) - 1j * np.concatenate([h, 2.0 * h]) * log_forward
    psi_h, psi_2h = psi[: h.size], psi[h.size :]
    ok = np.isfinite(psi_h) & np.isfinite(psi_2h) & (np.abs(psi_2h) <= 0.5)
    idx = int(np.flatnonzero(ok)[-1]) if ok.any() else 0
    step, p1, p2 = h[idx], psi_h[idx], psi_2h[idx]
    c4 = max(2.0 * (p2.real - 4.0 * p1.real) / step**4, 0.0)
    c2 = max(-2.0 * (p1.real - c4 * step**4 / 24.0) / step**2, 1e-12)
    c1 = log_forward + (8.0 * p1.imag - p2.imag) / (6.0 * step)
    half = width * math.sqrt(c2 + math.sqrt(c4))
    return c1 - half, c1 + half


def _chernoff_orders(edge: float) -> np.ndarray:
    """Moment orders below the strip edge, refined towards it (where the bound
    is tightest for exponentially decaying jump tails)."""
    if not math.isfinite(edge):
        return _CHERNOFF_ORDERS
    inside = _CHERNOFF_ORDERS[_CHERNOFF_ORDERS < edge]
    return np.unique(np.concatenate([inside, edge * (1.0 - 2.0 ** -np.arange(1, 16))]))


def _chernoff_reach(moments: np.ndarray, orders: np.ndarray, log_eps: float) -> float:
    """
    Smallest ``z`` with ``P(X > z) <= exp(log_eps)`` from ``P(X > z) <= E[e^(sX)]
    e^(-s z)``, given ``E[e^(sX)]`` at ``orders``. The scan stops at the first
    value that is not finite, real, positive and log-convex: past the edge of
    the moment strip the closed-form CFs are continued analytically and return
    garbage. ``nan`` if no order is usable.
    """
    m = moments
    with np.errstate(all="ignore"):
        ok = np.isfinite(m) & (np.abs(m.imag) <= 1e-8 * np.abs(m.real)) & (m.real > 0)
        log_m = np.log(np.where(ok, m.real, 1.0))
    n = int(np.flatnonzero(~ok)[0]) if (~ok).any() else m.size
    if n >= 3:
        slopes = np.diff(log_m[:n]) / np.diff(orders[:n])
        drop = np.flatnonzero(np.diff(slopes) < -1e-6 * (1.0 + np.abs(slopes[1:])))
        if drop.size:
            n = int(drop[0]) + 2
    if n == 0:
        return math.nan
    return float(np.min((log_m[:n] - log_eps) / orders[:n]))


def cos_truncation(
    phi: Any,
    log_forward: float,
    width: float = 10.0,
    tail_tol: float = 1e-12,
    strip: tuple[float, float] | None = None,
) -> tuple[float, float]:
    """
    Truncation range ``[a, b]`` of ``ln S_T``.

    Each side reaches until the Chernoff bound on the probability beyond it is
    below ``tail_tol``, using the exponential moments ``E[(S_T / F)^(+/-s)] =
    phi(-/+ i s) F^(-/+s)`` inside the model's moment ``strip`` (see
    `BaseModel.moment_strip`; scanned for validity if None). This follows
    jump-driven tails, which the cumulant rule of Fang & Oosterlee (2008) misses
    at short maturities (the standard deviation shrinks like ``sqrt(T)``, the
    tail decay rate does not). A side without usable moments falls back to the
    cumulant rule with ``width``.
    """
    lo, hi = strip if strip is not None else (-math.inf, math.inf)
    up, down = _chernoff_orders(hi), _chernoff_orders(-lo)
    with np.errstate(all="ignore"):
        values = np.asarray(phi(np.concatenate([-1j * up, 1j * down])), dtype=complex)
        upper = values[: up.size] * np.exp(-up * log_forward)
        lower = values[up.size :] * np.exp(down * log_forward)
    log_eps = math.log(tail_tol)
    reach_up = _chernoff_reach(upper, up, log_eps)
    reach_down = _chernoff_reach(lower, down, log_eps)
    if math.isnan(reach_up) or math.isnan(reach_down):
        a, b = _cumulant_range(phi, log_forward, width)
        if math.isnan(reach_up):
            reach_up = b - log_forward
        if math.isnan(reach_down):
            reach_down = log_forward - a
    return log_forward - reach_down, log_forward + reach_up


def cos_n_terms(
    phi: Any, a: float, b: float, *, tol: float = 1e-12, max_terms: int = 1 << 16
) -> int:
    """
    Smallest power of two ``N >= 64`` whose neglected tail, relative to the
    strike, is bounded by ``tol``: the put coefficients are ``O(K / u_k^2)``, so
    the tail sums to at most ``~ 2 |phi(u_N)| / (pi u_N)`` with
    ``u_N = N pi / (b - a)``. Capped at ``max_terms``; a characteristic function
    that decays only algebraically (Variance Gamma, CGMY with small ``Y`` at short
    maturities) can run to the cap.
    """
    n = _MIN_TERMS * 2 ** np.arange(int(math.log2(max_terms / _MIN_TERMS)) + 1)
    u = n * math.pi / (b - a)
    with np.errstate(all="ignore"):
        decay = np.abs(np.asarray(phi(u), dtype=complex))
    small = np.flatnonzero(2.0 * decay / (math.pi * u) < tol)
    return int(n[small[0]]) if small.size else int(max_terms)


def cos_prices(
    phi: Any,
    spot: float,
    strikes: np.ndarray,
    is_call: np.ndarray,
    r: float,
    q: float,
    T: float,
    *,
    width: float = 10.0,
    tail_tol: float = 1e-12,
    strip: tuple[float, float] | None = None,
    n_terms: int | None = None,
    tol: float = 1e-12,
    max_terms: int = 1 << 16,
) -> tuple[np.ndarray, np.ndarray]:
    """
    COS prices and deltas of European options on one maturity.

    Puts are expanded (bounded payoff, Fang & Oosterlee 2008, sec. 3.3) and
    calls follow from put-call parity. ``phi`` is the characteristic function
    of ``ln S_T``; the spot enters it only through ``exp(i u ln S)``, which gives
    the delta from the same series. ``strip`` is the model's moment strip (see
    `cos_truncation`). Returns ``(prices, deltas)``.
    """
    strikes = np.asarray(strikes, dtype=float)
    is_call = np.asarray(is_call, dtype=bool)
    log_forward = math.log(spot) + (r - q) * T
    a, b = cos_truncation(phi, log_forward, width, tail_tol, strip)
    n = n_terms or cos_n_terms(phi, a, b, tol=tol, max_terms=max_terms)
    u = np.arange(n) * (math.pi / (b - a))
    with np.errstate(all="ignore"):
        phase = np.asarray(phi(u), dtype=complex) * np.exp(-1j * u * a)
    # Put coefficients are O(K / u^2), so the neglected terms sum to at most
    # ~ 2 K |phi(u_N)| / (pi u_N); measured errors are 1e-4 to 4e-2 of it.
    tail_bound = 2.0 * strikes.max() * abs(phase[-1]) / (math.pi * u[-1])
    if n_terms is None and tail_bound > _TAIL_WARN * strikes.max():
        warnings.warn(
            f"COS series truncated at max_terms={max_terms}: the neglected tail "
            f"is bounded by {tail_bound:.1e} because the characteristic function "
            "decays slowly (a near-degenerate law, e.g. Variance Gamma or CGMY "
            "over a few days); the actual error is typically 100x smaller.",
            RuntimeWarning,
            stacklevel=2,
        )
    phase[0] *= 0.5

    # Put payoff (K - e^y)^+ on [a, d], d = ln K clipped to the range.
    d = np.clip(np.log(strikes), a, b)[:, None]
    arg = u * (d - a)
    cos_arg, sin_arg = np.cos(arg), np.sin(arg)
    exp_d = np.exp(d)
    chi = (cos_arg * exp_d - math.exp(a) + u * sin_arg * exp_d) / (1.0 + u * u)
    psi = np.empty_like(arg)
    psi[:, 0] = d[:, 0] - a
    psi[:, 1:] = sin_arg[:, 1:] / u[1:]
    coeffs = (2.0 / (b - a)) * (strikes[:, None] * psi - chi)

    df_r, df_q = math.exp(-r * T), math.exp(-q * T)
    put = df_r * (coeffs @ phase.real)
    put_delta = df_r * (coeffs @ (1j * u * phase).real) / spot
    prices = np.where(is_call, put + spot * df_q - strikes * df_r, put)
    deltas = np.where(is_call, put_delta + df_q, put_delta)
    return prices, deltas


class COSTechnique(BaseTechnique, GreekMixin, IVMixin):
    """
    Prices European options with the COS method (Fang & Oosterlee 2008).

    Works for every model with a characteristic function. The series converges
    exponentially for smooth densities and, unlike Gil-Pelaez quadrature on a
    fixed grid, stays accurate when the characteristic function decays slowly
    (strong Feller violation in Heston, Variance Gamma and CGMY at short
    maturities), because the number of terms adapts to the decay. The delta is
    a by-product of the pricing series.
    """

    def __init__(
        self,
        *,
        width: float = 10.0,
        n_terms: int | None = None,
        tol: float = 1e-12,
        max_terms: int = 1 << 16,
    ):
        """
        Parameters
        ----------
        width : float, optional
            Half-width ``L`` of the truncation range in units of
            ``sqrt(c2 + sqrt(c4))``, by default 10 (Fang & Oosterlee 2008).
        n_terms : int, optional
            Fixed number of cosine terms. By default it is chosen per option
            as the smallest power of two (at least 64) whose neglected tail is
            below ``tol`` (see `cos_n_terms`), capped at ``max_terms``.
        tol : float, optional
            Bound on the neglected tail relative to the strike, by default 1e-12.
        max_terms : int, optional
            Cap on the adaptive number of terms, by default 2**16.
        """
        if width <= 0:
            raise ValueError("width must be positive.")
        if n_terms is not None and n_terms < 2:
            raise ValueError("n_terms must be at least 2.")
        self.width = float(width)
        self.n_terms = None if n_terms is None else int(n_terms)
        self.tol = float(tol)
        self.max_terms = int(max_terms)
        self._cached_results: dict[str, float] = {}
        self._cache_key: tuple | None = None

    def _price_and_delta(
        self, option: Option, stock: Stock, model: BaseModel, rate: Rate, **kwargs: Any
    ) -> dict[str, float]:
        if not model.supports_cf:
            raise TypeError(
                f"Model '{model.name}' does not support a characteristic function."
            )
        S, K, T = stock.spot, option.strike, option.maturity
        r, q = rate.get_rate(T), stock.dividend
        phi = model.cf(t=T, spot=S, r=r, q=q, **kwargs)
        prices, deltas = cos_prices(
            phi,
            S,
            np.array([K]),
            np.array([option.option_type is OptionType.CALL]),
            r,
            q,
            T,
            width=self.width,
            strip=model.moment_strip(),
            n_terms=self.n_terms,
            tol=self.tol,
            max_terms=self.max_terms,
        )
        return {"price": float(prices[0]), "delta": float(deltas[0])}

    def price(
        self, option: Option, stock: Stock, model: BaseModel, rate: Rate, **kwargs: Any
    ) -> PricingResult:
        """
        Calculates the option price and caches the series delta.

        Parameters
        ----------
        option : Option
            The option contract to be priced.
        stock : Stock
            The underlying asset's properties.
        model : BaseModel
            The financial model to use. Must support a characteristic function.
        rate : Rate
            The risk-free rate structure.

        Returns
        -------
        PricingResult
            An object containing the calculated price.
        """
        self._cached_results = self._price_and_delta(
            option, stock, model, rate, **kwargs
        )
        self._cache_key = (option, stock, model, rate, kwargs)
        return PricingResult(price=self._cached_results["price"])

    def delta(
        self, option: Option, stock: Stock, model: BaseModel, rate: Rate, **kwargs: Any
    ) -> float:
        """
        Returns the delta computed with the price, pricing first if the inputs
        differ from the last `price` call.
        """
        try:
            cached = self._cache_key == (option, stock, model, rate, kwargs)
        except ValueError:  # array-valued kwargs have no truth value
            cached = False
        if not cached:
            self.price(option, stock, model, rate, **kwargs)
        delta_val = self._cached_results.get("delta")
        if delta_val is not None and np.isfinite(delta_val):
            return delta_val
        return super().delta(option, stock, model, rate, **kwargs)
