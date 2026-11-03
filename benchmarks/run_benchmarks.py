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
    FFTTechnique,
    IntegrationTechnique,
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


def _mc_bsm_100k() -> float:
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


CASES: dict[str, tuple[Callable[[], float], str]] = {
    "bsm_closed_form_x1000": (_bsm_closed_form, "1000 closed-form BSM prices"),
    "heston_fft": (_heston_fft, "Carr-Madan FFT, N=4096"),
    "heston_integration": (_heston_integration, "Gil-Pelaez quadrature"),
    "cn_pde_200x200": (_cn_pde, "Crank-Nicolson, 200 x 200 grid"),
    "mc_bsm_100k": (_mc_bsm_100k, "100k paths x 100 steps, antithetic"),
    "mc_heston_100k": (_mc_heston_100k, "100k paths x 100 steps, antithetic"),
    "lsmc_bsm_20k": (_lsmc_bsm_20k, "Longstaff-Schwartz, 20k paths x 50 steps"),
    "heston_chain_400": (
        _heston_chain_400,
        "vectorised chain pricer, 400 options x 5 maturities",
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--only", nargs="*", help="subset of case names")
    parser.add_argument("--out", type=Path, help="write JSON results to this path")
    args = parser.parse_args()

    results = run(args.repeats, args.only)
    payload = {
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "git": _git_rev(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "numba": numba.__version__,
        "results": results,
    }

    print(f"{'case':26s}{'median ms':>12s}{'min ms':>10s}   value")
    for name, r in results.items():
        print(f"{name:26s}{r['median_ms']:12.3f}{r['min_ms']:10.3f}   {r['value']:.6f}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
