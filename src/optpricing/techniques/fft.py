from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np

from optpricing.atoms import Option, OptionType, Rate, Stock
from optpricing.models import BaseModel
from optpricing.techniques.base import BaseTechnique, GreekMixin, IVMixin, PricingResult

__doc__ = """
Defines a pricing technique based on the Fast Fourier Transform (FFT) of the
dampened, risk-neutral option price, following the Carr-Madan formula.
"""

_ALPHA_STEP = 0.05
_ALPHA_SCAN = 120
_STRIP_MARGIN = 0.9
_DECAY_TOL = 1e-13
_TAIL_WARN = 1e-6
_MAX_POINTS = 1 << 18


_SCAN_OFFSETS = _ALPHA_STEP * np.arange(1, _ALPHA_SCAN)
_MID_MOMENTS = np.linspace(0.05, 0.95, 19)


def _moment_strip(a: np.ndarray, m: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Log-moments ``L(a) = log E[S_T^a]`` along a scan ``a`` moving away from the
    always-finite band ``0 <= a <= 1`` (``m = phi(-i a)``), cut at the edge of the
    strip where they are finite.

    Beyond the critical moment the closed-form CFs are continued analytically and
    return complex, negative or non-convex values (a pole of Kou's jump
    transform, a branch point of NIG/CGMY, Heston's moment explosion), so the
    scan stops at the first point that is not finite, real, positive and
    log-convex. Returns the admissible ``a``, ``L(a)`` and the edge (``inf`` if
    the scan only stopped on overflow or ran out).
    """
    with np.errstate(all="ignore"):
        real = np.isfinite(m) & (np.abs(m.imag) <= 1e-8 * np.abs(m.real))
        overflow = np.isinf(m.real) & (m.real > 0)
        ok = real & (m.real > 0)
        log_m = np.log(np.where(ok, m.real, 1.0))
    bad = np.flatnonzero(~ok)
    n = int(bad[0]) if bad.size else a.size
    edge = np.inf if n == a.size or overflow[n] else float(a[n])
    if n >= 3:
        curvature = np.diff(log_m[:n], 2)
        concave = np.flatnonzero(curvature < -1e-6 * (1.0 + np.abs(log_m[1 : n - 1])))
        if concave.size:
            n = int(concave[0]) + 1
            edge = float(a[n])
    return a[:n], log_m[:n], edge


def _lord_kahl_alpha(phi: Any, log_strike: float) -> tuple[float, float]:
    """
    Damping parameter of Lord & Kahl (2007): minimise
    ``-alpha k + log E[S_T^(alpha+1)] - log|alpha (alpha+1)|`` (the log of the
    damped integrand at ``v = 0``) over ``alpha > 0`` (the transform of the
    call), ``alpha < -1`` (the put) and ``-1 < alpha < 0`` (``C - S e^{-qT}``;
    always admissible, since ``E[S_T^a] < inf`` for ``0 < a < 1``), keeping
    ``alpha + 1`` inside 90 % of the moment strip.

    Returns ``alpha`` and the distance from ``alpha`` to the nearest singularity
    of the damped integrand (``alpha = 0``, ``alpha = -1`` or the strip edge),
    which sets the frequency resolution the quadrature needs.
    """
    up, down = 1.0 + _SCAN_OFFSETS, -_SCAN_OFFSETS
    with np.errstate(all="ignore"):
        moments = np.asarray(
            phi(-1j * np.concatenate([up, down, _MID_MOMENTS])), dtype=complex
        )
    n_scan = _SCAN_OFFSETS.size
    regions = []
    for a_start, direction, a_scan, m_scan in (
        (1.0, 1, up, moments[:n_scan]),
        (0.0, -1, down, moments[n_scan : 2 * n_scan]),
    ):
        a, log_m, edge = _moment_strip(a_scan, m_scan)
        if a.size >= 2:
            a_lim = a_start + _STRIP_MARGIN * (edge - a_start)
            regions.append((a, log_m, edge, direction * (a - a_lim) <= 0))
    with np.errstate(all="ignore"):
        log_mid = np.log(np.real(moments[2 * n_scan :]))
    regions.append((_MID_MOMENTS, log_mid, np.inf, np.isfinite(log_mid)))

    def objective(alpha: np.ndarray, log_m: np.ndarray) -> np.ndarray:
        with np.errstate(all="ignore"):
            values = -alpha * log_strike + log_m - np.log(np.abs(alpha * (alpha + 1.0)))
        return np.where(np.isfinite(values), values, np.inf)

    best: tuple[float, np.ndarray, int, float] | None = None
    for a, log_m, edge, admissible in regions:
        alpha = a - 1.0
        values = np.where(admissible, objective(alpha, log_m), np.inf)
        i = int(np.argmin(values))
        if np.isfinite(values[i]) and (best is None or values[i] < best[0]):
            best = (float(values[i]), alpha, i, edge)
    if best is None:
        raise ValueError("No admissible Carr-Madan damping parameter.")

    # Refine between the grid neighbours (one vectorised CF call) and take the
    # parabolic vertex, so alpha moves continuously with the inputs and
    # bump-and-reprice Greeks do not jump between grid points.
    _, alpha, i, edge = best
    lo, hi = alpha[max(i - 1, 0)], alpha[min(i + 1, alpha.size - 1)]
    fine = np.linspace(min(lo, hi), max(lo, hi), 41)
    with np.errstate(all="ignore"):
        log_fine = np.log(np.real(np.asarray(phi(-1j * (fine + 1.0)), dtype=complex)))
    values = objective(fine, log_fine)
    j = int(np.argmin(values))
    x = float(fine[j])
    if 0 < j < fine.size - 1 and np.all(np.isfinite(values[j - 1 : j + 2])):
        y0, y1, y2 = values[j - 1 : j + 2]
        curvature = y0 - 2.0 * y1 + y2
        if curvature > 0:
            x += 0.5 * (fine[1] - fine[0]) * (y0 - y2) / curvature
    return x, min(abs(x), abs(x + 1.0), abs(edge - (x + 1.0)))


class FFTTechnique(BaseTechnique, GreekMixin, IVMixin):
    """
    Fast Fourier Transform (FFT) pricer based on the Carr-Madan formula,
    preserving the original tuned logic for grid and parameter selection.
    """

    def __init__(
        self,
        *,
        n: int = 12,
        eta: float = 0.125,
        alpha: float | None = None,
    ):
        """
        Initializes the FFT solver.

        Parameters
        ----------
        n : int, optional
            The exponent for the minimum number of grid points (N = 2^n), by
            default 12. ``N`` is doubled (up to ``2^18``) until the damped
            characteristic function has decayed to 1e-12 at ``u = N eta``, and the
            log-strike grid is centred on the strike, so no interpolation is
            needed.
        eta : float, optional
            The spacing of the grid in the frequency domain, by default 0.125
            (0.25 left a 3e-5 discretisation error on ATM prices).
        alpha : float | None, optional
            The dampening parameter. If None, it is chosen per option by the
            Lord & Kahl (2007) rule inside the strip where ``E[S_T^(alpha+1)]``
            is finite (negative values below -1 price the put), and ``eta`` is
            reduced (with ``N`` grown to keep the log-strike spacing, up to
            ``2^18``) when ``alpha`` lies close to a singularity of the damped
            integrand. Defaults to None.
        """
        self.n = int(n)
        self.N = 1 << self.n
        self.base_eta = float(eta)
        self.alpha_user = alpha
        self._cached_results: dict[str, Any] = {}

    def _price_and_greeks(
        self,
        option: Option,
        stock: Stock,
        model: BaseModel,
        rate: Rate,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Internal method to perform the core FFT calculation once.

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
        """
        if not model.supports_cf:
            raise TypeError(
                f"Model '{model.name}' does not support a characteristic function."
            )

        S0, K, T = stock.spot, option.strike, option.maturity
        r, q = rate.get_rate(T), stock.dividend
        k_target = math.log(K)

        phi = model.cf(t=T, spot=S0, r=r, q=q, **kwargs)

        n_points = self.N
        if self.alpha_user is not None:
            alpha = self.alpha_user
            vol_proxy = self._get_vol_proxy(model, kwargs)
            eta = (
                self.base_eta * max(1.0, vol_proxy * math.sqrt(T))
                if vol_proxy is not None
                else self.base_eta
            )
        else:
            alpha, width = _lord_kahl_alpha(phi, k_target)
            # aliasing decays like exp(-width 2 pi / eta); width / 4 left 4e-7
            # next to Kou's pole, width / 8 leaves ~1e-11
            eta = min(self.base_eta, 0.125 * width)
            # keep the frequency range N eta of the default grid
            while n_points * eta < self.N * self.base_eta and n_points < _MAX_POINTS:
                n_points *= 2

        # Extend the frequency range until the neglected tail of the integrand
        # phi / u^2, ~ |phi(u_N)| / (pi u_N) relative to phi(-i(alpha+1)), is
        # negligible (a fixed u <= 512 truncated short maturities: 4e-2 at one day).
        shift = -1j * (alpha + 1.0)
        with np.errstate(all="ignore"):
            scale = abs(complex(np.asarray(phi(np.array([shift])))[0]))
            while True:
                u_max = n_points * eta
                edge = abs(complex(np.asarray(phi(np.array([u_max + shift])))[0]))
                tail = edge / (math.pi * u_max * scale)
                if tail <= _DECAY_TOL or n_points >= _MAX_POINTS:
                    break
                n_points *= 2
        if tail > _TAIL_WARN:
            warnings.warn(
                f"FFT frequency grid truncated at N={n_points}: the neglected tail "
                f"is ~{tail:.1e} of the spot because the characteristic function "
                "decays slowly (a near-degenerate law, e.g. Variance Gamma or CGMY "
                "over a few days); COSTechnique is faster there.",
                RuntimeWarning,
                stacklevel=3,
            )

        # Log-strike grid centred on the target, which is read off exactly.
        lambda_ = (2 * math.pi) / (n_points * eta)
        k_start = k_target - 0.5 * n_points * lambda_

        # Simpson's rule weights
        w = np.ones(n_points)
        w[1:-1:2], w[2:-2:2] = 4, 2
        weights = w * eta / 3.0

        u = np.arange(n_points) * eta
        discount = math.exp(-r * T)
        numerator = phi(u - 1j * (alpha + 1))
        denominator = alpha**2 + alpha - u**2 + 1j * u * (2 * alpha + 1)
        psi = discount * numerator / denominator

        fft_input = psi * np.exp(-1j * u * k_start) * weights
        fft_vals = np.fft.fft(fft_input).real

        call_price = math.exp(-alpha * k_target) / math.pi * fft_vals[n_points // 2]
        # The damped transform is that of C (alpha > 0), C - S e^{-qT}
        # (-1 < alpha < 0) or of the put (alpha < -1) (Lee 2004).
        if alpha < 0.0:
            call_price += S0 * math.exp(-q * T)
        if alpha < -1.0:
            call_price -= K * math.exp(-r * T)

        if option.option_type is OptionType.CALL:
            price = call_price
        else:
            price = call_price - (S0 * np.exp(-q * T) - K * np.exp(-r * T))

        return {"price": price}

    def price(
        self,
        option: Option,
        stock: Stock,
        model: BaseModel,
        rate: Rate,
        **kwargs: Any,
    ) -> PricingResult:
        """
        Calculates the option price using the FFT method.

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
        self._cached_results = self._price_and_greeks(
            option, stock, model, rate, **kwargs
        )
        return PricingResult(price=self._cached_results["price"])

    @staticmethod
    def _get_vol_proxy(
        model: BaseModel,
        kw: dict[str, Any],
    ) -> float | None:
        """
        Finds a best-effort volatility proxy from the model or kwargs.

        This is used for the heuristic auto-tuning of the FFT grid parameters.
        It checks for 'sigma', 'v0', and 'vol_of_vol' in that order.
        """
        if "sigma" in model.params and model.params["sigma"] is not None:
            return model.params["sigma"]
        if "v0" in kw and kw["v0"] is not None:
            return math.sqrt(kw["v0"])
        if "v0" in model.params and model.params["v0"] is not None:
            return math.sqrt(model.params["v0"])
        if "vol_of_vol" in model.params and model.params["vol_of_vol"] is not None:
            return model.params["vol_of_vol"]
        return None
