"""Micro-benchmarks for the core pricing paths.

Usage (from the repository root)::

    uv run python benchmarks/run_benchmarks.py                # print table
    uv run python benchmarks/run_benchmarks.py --out results/x.json
    uv run python benchmarks/run_benchmarks.py --repeats 9 --only mc_heston_100k

Each case is warmed up once (numba compilation, caches) and then timed
``repeats`` times; the median and minimum wall-clock times are reported.
Numbers are only comparable on the same machine, so results files record the
platform, interpreter and library versions alongside the git revision.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import numba
import numpy as np
import scipy

from optpricing import BSMModel, HestonModel, Option, OptionType, Rate, Stock
from optpricing.techniques import (
    AmericanMonteCarloTechnique,
    ClosedFormTechnique,
    CRRTechnique,
    FFTTechnique,
    IntegrationTechnique,
    LeisenReimerTechnique,
    MonteCarloTechnique,
    PDETechnique,
)

OPTION = Option(strike=100.0, maturity=1.0, option_type=OptionType.CALL)
PUT = Option(strike=100.0, maturity=1.0, option_type=OptionType.PUT)
STOCK = Stock(spot=100.0, dividend=0.01)
RATE = Rate(rate=0.05)
BSM = BSMModel(params={"sigma": 0.2})
HESTON = HestonModel(
    params={"v0": 0.04, "kappa": 2.0, "theta": 0.05, "rho": -0.7, "vol_of_vol": 0.5}
)


def _bsm_closed_form() -> float:
    tech = ClosedFormTechnique()
    for _ in range(1000):
        p = tech.price(OPTION, STOCK, BSM, RATE).price
    return p


def _heston_fft() -> float:
    return FFTTechnique().price(OPTION, STOCK, HESTON, RATE, v0=0.04).price


def _heston_integration() -> float:
    return IntegrationTechnique().price(OPTION, STOCK, HESTON, RATE, v0=0.04).price


def _cn_pde() -> float:
    return PDETechnique(M=200, N=200).price(OPTION, STOCK, BSM, RATE).price


def _crr_am_1000() -> float:
    return CRRTechnique(steps=1000, is_american=True).price(PUT, STOCK, BSM, RATE).price


def _crr_am_5000() -> float:
    return CRRTechnique(steps=5000, is_american=True).price(PUT, STOCK, BSM, RATE).price


def _lr_am_5000() -> float:
    # Even step counts are bumped to the next odd number inside the kernel.
    return (
        LeisenReimerTechnique(steps=5000, is_american=True)
        .price(PUT, STOCK, BSM, RATE)
        .price
    )


def _mc_bsm_100k() -> float:
    # Black-Scholes has no exact terminal sampler here: this is Euler on log S,
    # one draw per step, so the 100 steps are real work.
    mc = MonteCarloTechnique(n_paths=100_000, n_steps=100, seed=1)
    return mc.price(OPTION, STOCK, BSM, RATE).price


def _mc_heston_100k() -> float:
    mc = MonteCarloTechnique(n_paths=100_000, n_steps=100, seed=1)
    return mc.price(OPTION, STOCK, HESTON, RATE, v0=0.04).price


def _lsmc_bsm_20k() -> float:
    amc = AmericanMonteCarloTechnique(n_paths=20_000, n_steps=50, seed=1)
    return amc.price(PUT, STOCK, BSM, RATE).price


def _heston_chain_400() -> float:
    import pandas as pd

    from optpricing.calibration.vectorized_pricer import price_options_vectorized

    chain = pd.DataFrame(
        [
            {"strike": k, "maturity": t, "optionType": typ}
            for t in (0.1, 0.25, 0.5, 1.0, 2.0)
            for k in np.linspace(60, 150, 40)
            for typ in ("call", "put")
        ]
    )
    return float(price_options_vectorized(chain, STOCK, HESTON, RATE, v0=0.04).sum())


# Spot 100, strike 100, one year, rate 5%, dividend 1%, unless the row says otherwise.
CASES: dict[str, tuple[Callable[[], float], str]] = {
    "bsm_closed_form_x1000": (
        _bsm_closed_form,
        "1,000 European calls, closed form, σ = 0.2",
    ),
    "heston_fft": (_heston_fft, "one Heston call, Carr–Madan FFT, default grid"),
    "heston_integration": (
        _heston_integration,
        "one Heston call, Gil–Pelaez quadrature",
    ),
    "cn_pde_200x200": (
        _cn_pde,
        "one European call, Crank–Nicolson, 200 spot × 200 time steps",
    ),
    "crr_am_1000": (_crr_am_1000, "American put, CRR, 1,000 steps"),
    "crr_am_5000": (_crr_am_5000, "American put, CRR, 5,000 steps"),
    "lr_am_5000": (_lr_am_5000, "American put, Leisen–Reimer, 5,001 steps"),
    "mc_bsm_100k": (
        _mc_bsm_100k,
        "European call, Euler on log S, 100k paths × 100 steps, antithetic",
    ),
    "mc_heston_100k": (
        _mc_heston_100k,
        "Heston call, Andersen QE, 100k paths × 100 steps, antithetic",
    ),
    "lsmc_bsm_20k": (
        _lsmc_bsm_20k,
        "American put, Longstaff–Schwartz degree 2, 20k paths × 50 steps",
    ),
    "heston_chain_400": (
        _heston_chain_400,
        "400 Heston options, COS, 5 maturities × 40 strikes × call and put",
    ),
}


def _git_rev() -> str:
    try:
        rev = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        ).stdout.strip()
        return rev + ("-dirty" if dirty else "")
    except Exception:  # pragma: no cover - git may be unavailable
        return "unknown"


def run(repeats: int, only: list[str] | None) -> dict:
    results = {}
    for name, (fn, desc) in CASES.items():
        if only and name not in only:
            continue
        value = fn()  # warm-up: numba compile, lazy imports
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            fn()
            times.append(time.perf_counter() - t0)
        results[name] = {
            "description": desc,
            "median_ms": 1e3 * statistics.median(times),
            "min_ms": 1e3 * min(times),
            "repeats": repeats,
            "value": float(value),
        }
    return results


def _ql_cell(row: dict) -> str:
    value = row.get("quantlib_median_ms")
    if isinstance(value, int | float):
        return f"{value:.2f}"
    return "—"


def markdown_table(payload: dict) -> str:
    """README block: one machine and revision, then case, contract, medians."""
    sample = next(iter(payload["results"].values()))
    intro = (
        f"Medians of {sample['repeats']} runs on `{payload['platform']}`, "
        f"git `{payload['git']}`, Python {payload['python']}, "
        f"NumPy {payload['numpy']}, SciPy {payload['scipy']}, "
        f"Numba {payload['numba']}."
    )
    quantlib = payload.get("quantlib")
    if quantlib:
        intro += f" QuantLib {quantlib} times the same contract with its own engine."
    lines = [
        intro,
        "",
        "| Case | Prices | Median (ms) | QuantLib (ms) |",
        "| --- | --- | ---: | ---: |",
    ]
    for name, row in payload["results"].items():
        lines.append(
            f"| `{name}` | {row['description']} | {row['median_ms']:.2f} | "
            f"{_ql_cell(row)} |"
        )
    return "\n".join(lines)


def _quantlib_medians(repeats: int) -> tuple[str, dict[str, float]] | None:
    """QuantLib 1.44 timings for the same contracts. None if QuantLib is absent.

    The two single-option Heston rows share one ``AnalyticHestonEngine`` timing.
    Leisen–Reimer is run at 5,001 steps, which is what this library's kernel does
    with a request for 5,000. Monte Carlo uses the same path and step counts,
    antithetic draws, and seed 1.
    """
    try:
        import QuantLib as ql  # noqa: N813 — the QuantLib tutorial name
    except ImportError:
        return None

    today = ql.Date(18, 11, 2026)
    ql.Settings.instance().evaluationDate = today
    day = ql.Actual365Fixed()
    spot = ql.QuoteHandle(ql.SimpleQuote(100.0))
    risk_free = ql.YieldTermStructureHandle(ql.FlatForward(today, 0.05, day))
    dividend = ql.YieldTermStructureHandle(ql.FlatForward(today, 0.01, day))
    vol = ql.BlackVolTermStructureHandle(
        ql.BlackConstantVol(today, ql.NullCalendar(), 0.20, day)
    )
    bsm = ql.BlackScholesMertonProcess(spot, dividend, risk_free, vol)
    expiry = today + 365
    european = ql.EuropeanExercise(expiry)
    american = ql.AmericanExercise(today, expiry)
    call = ql.PlainVanillaPayoff(ql.Option.Call, 100.0)
    put = ql.PlainVanillaPayoff(ql.Option.Put, 100.0)
    heston = ql.HestonProcess(risk_free, dividend, spot, 0.04, 2.0, 0.05, 0.5, -0.7)
    heston_engine = ql.AnalyticHestonEngine(ql.HestonModel(heston), 1e-12, 100_000)

    def european_call() -> float:
        # NPV() caches an unchanged option, so each of the 1,000 prices has to
        # clear that cache. A fresh VanillaOption per price would time construction.
        option = ql.VanillaOption(call, european)
        option.setPricingEngine(ql.AnalyticEuropeanEngine(bsm))
        price = 0.0
        for _ in range(1000):
            option.recalculate()
            price = option.NPV()
        return price

    def heston_analytic() -> float:
        option = ql.VanillaOption(call, european)
        option.setPricingEngine(heston_engine)
        return option.NPV()

    def pde() -> float:
        option = ql.VanillaOption(call, european)
        option.setPricingEngine(ql.FdBlackScholesVanillaEngine(bsm, 200, 200))
        return option.NPV()

    def binomial(kind: str, steps: int) -> Callable[[], float]:
        def price() -> float:
            option = ql.VanillaOption(put, american)
            option.setPricingEngine(ql.BinomialVanillaEngine(bsm, kind, steps))
            return option.NPV()

        return price

    def mc_bsm() -> float:
        option = ql.VanillaOption(call, european)
        option.setPricingEngine(
            ql.MCEuropeanEngine(
                bsm,
                "PseudoRandom",
                timeSteps=100,
                requiredSamples=100_000,
                seed=1,
                antitheticVariate=True,
            )
        )
        return option.NPV()

    def mc_heston() -> float:
        option = ql.VanillaOption(call, european)
        option.setPricingEngine(
            ql.MCEuropeanHestonEngine(
                heston,
                "PseudoRandom",
                timeSteps=100,
                requiredSamples=100_000,
                seed=1,
                antitheticVariate=True,
            )
        )
        return option.NPV()

    def lsmc() -> float:
        option = ql.VanillaOption(put, american)
        option.setPricingEngine(
            ql.MCAmericanEngine(
                bsm,
                "PseudoRandom",
                timeSteps=50,
                requiredSamples=20_000,
                seed=1,
                antitheticVariate=True,
                polynomOrder=2,
            )
        )
        return option.NPV()

    def chain() -> float:
        total = 0.0
        for maturity in (0.1, 0.25, 0.5, 1.0, 2.0):
            exercise = ql.EuropeanExercise(today + int(round(maturity * 365)))
            for strike in np.linspace(60, 150, 40):
                for kind in (ql.Option.Call, ql.Option.Put):
                    option = ql.VanillaOption(
                        ql.PlainVanillaPayoff(kind, float(strike)), exercise
                    )
                    option.setPricingEngine(heston_engine)
                    total += option.NPV()
        return total

    jobs: dict[str, Callable[[], float]] = {
        "bsm_closed_form_x1000": european_call,
        "heston_analytic": heston_analytic,
        "cn_pde_200x200": pde,
        "crr_am_1000": binomial("crr", 1000),
        "crr_am_5000": binomial("crr", 5000),
        "lr_am_5000": binomial("lr", 5001),
        "mc_bsm_100k": mc_bsm,
        "mc_heston_100k": mc_heston,
        "lsmc_bsm_20k": lsmc,
        "heston_chain_400": chain,
    }
    medians: dict[str, float] = {}
    for name, fn in jobs.items():
        fn()
        samples = []
        for _ in range(repeats):
            started = time.perf_counter()
            fn()
            samples.append(time.perf_counter() - started)
        medians[name] = 1e3 * statistics.median(samples)
    shared = medians.pop("heston_analytic")
    medians["heston_fft"] = shared
    medians["heston_integration"] = shared
    return ql.__version__, medians


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--only", nargs="*", help="subset of case names")
    parser.add_argument("--out", type=Path, help="write JSON results to this path")
    args = parser.parse_args()

    results = run(args.repeats, args.only)
    quantlib = _quantlib_medians(args.repeats) if not args.only else None
    if quantlib is None and not args.only:
        print(
            "QuantLib is not importable, so the comparison column is empty. "
            "Install QuantLib 1.44 to fill it.",
            file=sys.stderr,
        )
    quantlib_version = None
    if quantlib is not None:
        quantlib_version, quantlib_times = quantlib
        for name, median in quantlib_times.items():
            if name in results:
                results[name]["quantlib_median_ms"] = median
    payload = {
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "git": _git_rev(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "numba": numba.__version__,
        "quantlib": quantlib_version,
        "results": results,
    }

    print(f"{'case':26s}{'median ms':>12s}{'min ms':>10s}   value")
    for name, r in results.items():
        print(f"{name:26s}{r['median_ms']:12.3f}{r['min_ms']:10.3f}   {r['value']:.6f}")
    print()
    print(markdown_table(payload))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
