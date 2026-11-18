"""Reproduce the validation tables in the README and the user guide.

The cases are the reference contracts in ``tests/correctness/``. QuantLib is
not imported: the numbers pinned in those tests are the oracle.

::

    uv run python benchmarks/accuracy.py            # print the tables
    uv run python benchmarks/accuracy.py --write    # refresh JSON and the docs
    uv run python benchmarks/accuracy.py --check    # reprice and compare

``--check`` allows a small relative gap so an Ubuntu runner can confirm the
committed table without requiring bit-identical floats.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JSON_PATH = ROOT / "benchmarks" / "accuracy.json"
README_PATH = ROOT / "README.md"
GUIDE_PATH = ROOT / "docs" / "guide" / "validation.md"

SUMMARY = "accuracy-summary"
SUMMARY_FULL = "accuracy-summary-full"
CONVERGENCE = "accuracy-convergence"

QL_HESTON = "QuantLib 1.44 AnalyticHestonEngine"
QL_BATES = "QuantLib 1.44 BatesEngine"
QL_CEV = "QuantLib 1.44 AnalyticCEVEngine"
QL_BOND = "QuantLib 1.44 discount bond"
QL_AMERICAN = "QuantLib 1.44 FdBlackScholesVanillaEngine"
BSM_CF = "BSM closed form"
COS_REF = "COSTechnique"
FO2008 = "Fang and Oosterlee (2008)"
COS_PAPER = "COS, N=2^18, on the paper characteristic function"
COS_SHORT = "COS, N=2^18 (Fang and Oosterlee truncation)"
COS_SLOW = "COS, N=2^22, fixed range"
MERTON_SERIES = "Merton (1976) series, 80 terms"
HULL_BLACK = "Hull, Black's approximation"
HULL_CEV = "Hull CEV formula (zero rates also QuantLib 1.44)"
MERTON_PUT = "Merton (1973) perpetual put"
QUAD_REF = "Gil-Pelaez quadrature"


@dataclass(frozen=True)
class Obs:
    model: str
    technique: str
    reference: str
    family: str
    label: str
    error: float


def format_error(value: float) -> str:
    if not math.isfinite(value):
        return "inf"
    if value == 0.0:
        return "0"
    return f"{value:.1e}"


def format_slope(value: float) -> str:
    return f"{value:.2f}"


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _err(price: float, ref: float) -> float:
    if not math.isfinite(price) or not math.isfinite(ref):
        return math.inf
    return abs(float(price) - float(ref))


def _price(technique, option, stock, model, rate, **kwargs) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        warnings.simplefilter("ignore", RuntimeWarning)
        return float(technique.price(option, stock, model, rate, **kwargs).price)


def _load_cases():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from tests.correctness._helpers import make_option
    from tests.correctness.conftest import cf_model_cases
    from tests.correctness.test_convergence import _errors, _slopes
    from tests.correctness.test_reference_values import (
        AMERICAN_PUT_CASES,
        BATES_CASES,
        CEV_ABOVE_ONE,
        CEV_ABOVE_ONE_CALLS,
        CEV_CASES,
        CEV_NEAR_LOGNORMAL,
        FELLER_CASES,
        FFT_DAMPING_CASES,
        HESTON_CASES,
        HULL_DIVIDENDS,
        LEVY_GOLDEN,
        MERTON_CASES,
        SHORT_MATURITY_CASES,
        SLOW_CF_CASES,
        STRONG_LEVERAGE_CASES,
        ZCB_CASES,
        _fft_damping_case,
        _levy_golden_model,
        _short_maturity_model,
        _slow_cf_model,
    )

    return {
        "make_option": make_option,
        "cf_models": cf_model_cases,
        "errors": _errors,
        "slopes": _slopes,
        "AMERICAN_PUT_CASES": AMERICAN_PUT_CASES,
        "BATES_CASES": BATES_CASES,
        "CEV_ABOVE_ONE": CEV_ABOVE_ONE,
        "CEV_ABOVE_ONE_CALLS": CEV_ABOVE_ONE_CALLS,
        "CEV_CASES": CEV_CASES,
        "CEV_NEAR_LOGNORMAL": CEV_NEAR_LOGNORMAL,
        "FELLER_CASES": FELLER_CASES,
        "FFT_DAMPING_CASES": FFT_DAMPING_CASES,
        "HESTON_CASES": HESTON_CASES,
        "HULL_DIVIDENDS": HULL_DIVIDENDS,
        "LEVY_GOLDEN": LEVY_GOLDEN,
        "MERTON_CASES": MERTON_CASES,
        "SHORT_MATURITY_CASES": SHORT_MATURITY_CASES,
        "SLOW_CF_CASES": SLOW_CF_CASES,
        "STRONG_LEVERAGE_CASES": STRONG_LEVERAGE_CASES,
        "ZCB_CASES": ZCB_CASES,
        "fft_damping_case": _fft_damping_case,
        "levy_golden_model": _levy_golden_model,
        "short_maturity_model": _short_maturity_model,
        "slow_cf_model": _slow_cf_model,
    }


def _observe(rows: list[Obs], **kwargs) -> None:
    rows.append(Obs(**kwargs))


def collect() -> tuple[list[Obs], list[dict]]:
    import pandas as pd

    from optpricing import (
        AmericanMonteCarloTechnique,
        BlacksApproxModel,
        BSMModel,
        CEVModel,
        CGMYModel,
        ClosedFormTechnique,
        COSTechnique,
        CRRTechnique,
        FFTTechnique,
        IntegrationTechnique,
        LeisenReimerTechnique,
        MonteCarloTechnique,
        PDETechnique,
        PerpetualPutModel,
        Rate,
        Stock,
        TOPMTechnique,
        VarianceGammaModel,
        ZeroCouponBond,
    )

    c = _load_cases()
    make_option = c["make_option"]
    rows: list[Obs] = []
    closed = ClosedFormTechnique()
    cos = COSTechnique()
    fft = FFTTechnique()
    quad = IntegrationTechnique()

    _log("Heston / Bates / Merton reference cases")
    for case in c["HESTON_CASES"]:
        label, s, k, t, r, q, v0, kappa, theta, xi, rho, call, ref = case
        model = _heston(v0, kappa, theta, xi, rho)
        option = make_option(k, t, call)
        stock, rate = Stock(spot=s, dividend=q), Rate(rate=r)
        for name, tech in (("COS", cos), ("Gauss-Legendre", quad), ("FFT", fft)):
            got = _price(tech, option, stock, model, rate, v0=v0)
            _observe(
                rows,
                model="Heston",
                technique=name,
                reference=QL_HESTON,
                family="quantlib-analytic",
                label=label,
                error=_err(got, ref),
            )

    for name, t, k, ref in c["SHORT_MATURITY_CASES"]:
        model = c["short_maturity_model"](name)
        option = make_option(k, t)
        stock, rate = Stock(spot=100.0, dividend=0.02), Rate(rate=0.05)
        reference = QL_HESTON if name == "heston" else COS_SHORT
        family = "quantlib-analytic" if name == "heston" else "literature"
        model_name = {"heston": "Heston", "nig": "NIG", "hyperbolic": "Hyperbolic"}[
            name
        ]
        for tech_name, tech in (("COS", cos), ("Gauss-Legendre", quad)):
            got = _price(tech, option, stock, model, rate)
            _observe(
                rows,
                model=model_name,
                technique=tech_name,
                reference=reference,
                family=family,
                label=f"T={t:.6g} K={k:g}",
                error=_err(got, ref),
            )
        chain_df = pd.DataFrame(
            {"strike": [k], "maturity": [t], "optionType": ["call"]}
        )
        from optpricing.calibration.vectorized_pricer import price_options_vectorized

        chain_methods = (
            ("cos", "chain COS"),
            ("gauss", "chain Gauss-Legendre"),
        )
        for method, tech_name in chain_methods:
            got = float(
                price_options_vectorized(chain_df, stock, model, rate, method=method)[0]
            )
            _observe(
                rows,
                model=model_name,
                technique=tech_name,
                reference=reference,
                family=family,
                label=f"T={t:.6g} K={k:g}",
                error=_err(got, ref),
            )

    for name, k, call, ref in c["LEVY_GOLDEN"]:
        model = c["levy_golden_model"](name)
        option = make_option(k, 1.0, call)
        stock, rate = Stock(spot=100.0, dividend=0.02), Rate(rate=0.05)
        model_name = {"kou": "Kou", "nig": "NIG", "hyperbolic": "Hyperbolic"}[name]
        for tech_name, tech in (("COS", cos), ("Gauss-Legendre", quad)):
            got = _price(tech, option, stock, model, rate)
            _observe(
                rows,
                model=model_name,
                technique=tech_name,
                reference=COS_PAPER,
                family="literature",
                label=f"K={k:g} {'call' if call else 'put'}",
                error=_err(got, ref),
            )

    # Same three contracts as test_levy_models_match_fang_oosterlee_2008.
    # They are written inside the test decorator, not as a module constant.
    fo_cases = [
        (
            "VG",
            VarianceGammaModel(params={"sigma": 0.12, "nu": 0.2, "theta": -0.14}),
            90.0,
            19.099354724,
        ),
        (
            "CGMY Y=0.5",
            CGMYModel(params={"C": 1.0, "G": 5.0, "M": 5.0, "Y": 0.5}),
            100.0,
            19.812948843,
        ),
        (
            "CGMY Y=1.5",
            CGMYModel(params={"C": 1.0, "G": 5.0, "M": 5.0, "Y": 1.5}),
            100.0,
            49.790905469,
        ),
    ]
    stock_fo, rate_fo = Stock(spot=100.0), Rate(rate=0.1)
    for label, model, k, ref in fo_cases:
        option = make_option(k, 1.0)
        model_name = "VG" if label == "VG" else "CGMY"
        for tech_name, tech in (("COS", cos), ("Gauss-Legendre", quad)):
            got = _price(tech, option, stock_fo, model, rate_fo)
            _observe(
                rows,
                model=model_name,
                technique=tech_name,
                reference=FO2008,
                family="literature",
                label=label,
                error=_err(got, ref),
            )

    _log("FFT damping, Feller, slow characteristic functions, leverage")
    for name, k, ref in c["FFT_DAMPING_CASES"]:
        model, r, q = c["fft_damping_case"](name)
        got = _price(
            fft, make_option(k, 1.0), Stock(spot=100.0, dividend=q), model, Rate(rate=r)
        )
        model_name = {
            "kou": "Kou",
            "cgmy": "CGMY",
            "nig": "NIG",
            "vg": "VG",
        }.get(name.split("_", 1)[0], name.split("_", 1)[0])
        _observe(
            rows,
            model=model_name,
            technique="FFT",
            reference=COS_PAPER,
            family="literature",
            label=f"{name} K={k:g}",
            error=_err(got, ref),
        )

    feller_params = {
        "v0": 0.005,
        "kappa": 1.0,
        "theta": 0.015625,
        "rho": -0.9375,
        "vol_of_vol": 1.0,
    }
    for k, ref in c["FELLER_CASES"]:
        model = _heston(**feller_params)
        option = make_option(k, 91 / 365)
        for tech_name, tech in (("COS", cos), ("Gauss-Legendre", quad)):
            got = _price(
                tech, option, Stock(spot=100.0), model, Rate(rate=0.0), v0=0.005
            )
            _observe(
                rows,
                model="Heston",
                technique=tech_name,
                reference=QL_HESTON,
                family="quantlib-analytic",
                label=f"Feller K={k:g}",
                error=_err(got, ref),
            )

    for name, k, ref in c["SLOW_CF_CASES"]:
        got = _price(
            cos,
            make_option(k, 0.05),
            Stock(spot=100.0, dividend=0.02),
            c["slow_cf_model"](name),
            Rate(rate=0.04),
        )
        _observe(
            rows,
            model=name.upper() if name == "vg" else "CGMY",
            technique="COS",
            reference=COS_SLOW,
            family="literature",
            label=f"T=0.05 K={k:g}",
            error=_err(got, ref),
        )

    for name, kappa, k, call, ref in c["STRONG_LEVERAGE_CASES"]:
        params = {
            "v0": 0.05,
            "kappa": kappa,
            "theta": 0.25,
            "rho": 0.5,
            "vol_of_vol": 1.0,
        }
        if name == "heston":
            model = _heston(**params)
            model_name, reference = "Heston", QL_HESTON
        else:
            from optpricing import BatesModel

            model = BatesModel(
                params={**params, "lambda": 0.5, "mu_j": -0.1, "sigma_j": 0.15}
            )
            model_name, reference = "Bates", QL_BATES
        option = make_option(k, 1.0, call)
        stock, rate = Stock(spot=100.0, dividend=0.01), Rate(rate=0.03)
        for tech_name, tech in (("COS", cos), ("Gauss-Legendre", quad)):
            got = _price(tech, option, stock, model, rate, v0=0.05)
            _observe(
                rows,
                model=model_name,
                technique=tech_name,
                reference=reference,
                family="quantlib-analytic",
                label=f"kappa={kappa:g} K={k:g}",
                error=_err(got, ref),
            )

    for case in c["MERTON_CASES"]:
        label, s, k, t, r, q, sigma, lam, mu_j, sig_j, call, ref = case
        from optpricing import MertonJumpModel

        model = MertonJumpModel(
            params={
                "sigma": sigma,
                "lambda": lam,
                "mu_j": mu_j,
                "sigma_j": sig_j,
                "max_sum_terms": 100,
            }
        )
        option = make_option(k, t, call)
        stock, rate = Stock(spot=s, dividend=q), Rate(rate=r)
        for tech_name, tech in (("closed form", closed), ("Gauss-Legendre", quad)):
            got = _price(tech, option, stock, model, rate)
            _observe(
                rows,
                model="Merton",
                technique=tech_name,
                reference=MERTON_SERIES,
                family="literature",
                label=label,
                error=_err(got, ref),
            )

    from optpricing import MertonJumpModel

    busy = MertonJumpModel(
        params={
            "sigma": 0.1,
            "lambda": 120.0,
            "mu_j": -0.002,
            "sigma_j": 0.01,
            "max_sum_terms": 600,
        }
    )
    busy_args = (
        make_option(100.0, 2.0),
        Stock(spot=100.0, dividend=0.01),
        busy,
        Rate(rate=0.04),
    )
    closed_busy = _price(closed, *busy_args)
    quad_busy = _price(quad, *busy_args)
    _observe(
        rows,
        model="Merton",
        technique="closed form",
        reference=QUAD_REF,
        family="internal",
        label="lambda T = 240",
        error=_err(closed_busy, quad_busy),
    )

    for case in c["BATES_CASES"]:
        (
            label,
            s,
            k,
            t,
            r,
            q,
            v0,
            kappa,
            theta,
            xi,
            rho,
            lam,
            mu_j,
            sig_j,
            call,
            ref,
        ) = case
        from optpricing import BatesModel

        model = BatesModel(
            params={
                "v0": v0,
                "kappa": kappa,
                "theta": theta,
                "rho": rho,
                "vol_of_vol": xi,
                "lambda": lam,
                "mu_j": mu_j,
                "sigma_j": sig_j,
            }
        )
        got = _price(
            quad,
            make_option(k, t, call),
            Stock(spot=s, dividend=q),
            model,
            Rate(rate=r),
            v0=v0,
        )
        _observe(
            rows,
            model="Bates",
            technique="Gauss-Legendre",
            reference=QL_BATES,
            family="quantlib-analytic",
            label=label,
            error=_err(got, ref),
        )

    _log("bonds, Black, perpetual put, CEV, American, trees")
    for case in c["ZCB_CASES"]:
        label, model_cls, r0, kappa, theta, sigma, t, ref = case
        model = model_cls(params={"kappa": kappa, "theta": theta, "sigma": sigma})
        model_name = "Vasicek" if "vasicek" in label else "CIR"
        for face in (1.0, 100.0):
            got = _price(
                closed,
                ZeroCouponBond(maturity=t, face_value=face),
                Stock(spot=r0),
                model,
                Rate(rate=0.0),
            )
            _observe(
                rows,
                model=model_name,
                technique="closed form",
                reference=QL_BOND,
                family="quantlib-analytic",
                label=f"{label} face={face:g}",
                error=_err(got / face, ref),
            )

    divs, times = c["HULL_DIVIDENDS"]
    for order in (1, -1):
        ordered = (divs[::order], times[::order])
        for via in ("stock", "kwargs"):
            if via == "stock":
                stock = Stock(
                    spot=40.0, discrete_dividends=ordered[0], ex_div_times=ordered[1]
                )
                kwargs = {}
            else:
                stock = Stock(spot=40.0)
                kwargs = {
                    "discrete_dividends": ordered[0],
                    "ex_div_times": ordered[1],
                }
            got = _price(
                closed,
                make_option(40.0, 0.5),
                stock,
                BlacksApproxModel(params={"sigma": 0.3}),
                Rate(rate=0.09),
                **kwargs,
            )
            _observe(
                rows,
                model="Black's approximation",
                technique="closed form",
                reference=HULL_BLACK,
                family="literature",
                label=f"{via} order={order}",
                error=_err(got, 3.671233209),
            )

    k_put = 100.0
    for sigma, r in ((0.2, 0.08), (0.4, 0.03)):
        s_star = 2 * r * k_put / (2 * r + sigma**2)
        model = PerpetualPutModel(params={"sigma": sigma, "rate": r})
        for spot in (s_star * 0.9, s_star * 1.2, 120.0, 200.0):
            if spot <= s_star:
                ref = k_put - spot
            else:
                ref = (k_put - s_star) * (spot / s_star) ** (-2 * r / sigma**2)
            got = model.price_closed_form(spot=spot, strike=k_put, q=0.0)
            _observe(
                rows,
                model="Perpetual put",
                technique="closed form",
                reference=MERTON_PUT,
                family="literature",
                label=f"sigma={sigma:g} S={spot:.4g}",
                error=_err(got, ref),
            )

    for case in c["CEV_CASES"]:
        label, s, k, t, r, q, sigma, gamma, ref = case
        model = CEVModel(params={"sigma": sigma, "gamma": gamma})
        got = _price(
            closed, make_option(k, t), Stock(spot=s, dividend=q), model, Rate(rate=r)
        )
        _observe(
            rows,
            model="CEV",
            technique="closed form",
            reference=HULL_CEV,
            family="literature",
            label=label,
            error=_err(got, ref),
        )
        mc = MonteCarloTechnique(n_paths=400_000, seed=11)
        mc_price = _price(
            mc, make_option(k, t), Stock(spot=s, dividend=q), model, Rate(rate=r)
        )
        _observe(
            rows,
            model="CEV",
            technique="Monte Carlo",
            reference=HULL_CEV,
            family="monte-carlo",
            label=label,
            error=_err(mc_price, ref),
        )

    for gamma, t, k, ref in c["CEV_NEAR_LOGNORMAL"]:
        model = CEVModel(
            params={"sigma": 0.25 * 100.0 ** (1.0 - gamma), "gamma": gamma}
        )
        got = _price(
            closed,
            make_option(k, t, call=False),
            Stock(spot=100.0, dividend=0.02),
            model,
            Rate(rate=0.05),
        )
        _observe(
            rows,
            model="CEV",
            technique="closed form",
            reference=QL_CEV,
            family="quantlib-analytic",
            label=f"gamma={gamma} K={k:g}",
            error=_err(got, ref),
        )

    for gamma, sigma, t, k, ref in c["CEV_ABOVE_ONE"]:
        got = _price(
            closed,
            make_option(k, t, call=False),
            Stock(spot=100.0, dividend=0.02),
            CEVModel(params={"sigma": sigma, "gamma": gamma}),
            Rate(rate=0.05),
        )
        _observe(
            rows,
            model="CEV",
            technique="closed form",
            reference=QL_CEV,
            family="quantlib-analytic",
            label=f"put gamma={gamma:g} K={k:g} T={t:.4g}",
            error=_err(got, ref),
        )

    for gamma, sigma, t, k, ref in c["CEV_ABOVE_ONE_CALLS"]:
        got = _price(
            closed,
            make_option(k, t),
            Stock(spot=100.0, dividend=0.02),
            CEVModel(params={"sigma": sigma, "gamma": gamma}),
            Rate(rate=0.05),
        )
        _observe(
            rows,
            model="CEV",
            technique="closed form",
            reference=QL_CEV,
            family="quantlib-analytic",
            label=f"call gamma={gamma:g} K={k:g} T={t:.4g}",
            error=_err(got, ref),
        )

    bsm = BSMModel(params={"sigma": 0.2})
    euro_stock, euro_rate = Stock(spot=100.0, dividend=0.01), Rate(rate=0.05)
    euro = make_option(100.0, 1.0)
    euro_ref = _price(closed, euro, euro_stock, bsm, euro_rate)
    for tech_name, tech in (
        ("CRR", CRRTechnique(steps=1000)),
        ("Leisen-Reimer", LeisenReimerTechnique(steps=501)),
        ("TOPM", TOPMTechnique(steps=1000)),
        ("Crank-Nicolson", PDETechnique(M=200, N=200)),
    ):
        got = _price(tech, euro, euro_stock, bsm, euro_rate)
        _observe(
            rows,
            model="BSM",
            technique=tech_name,
            reference=BSM_CF,
            family="internal",
            label="S=K=100",
            error=_err(got, euro_ref),
        )

    for case in c["AMERICAN_PUT_CASES"]:
        label, s, k, t, r, q, sigma, ref = case
        option = make_option(k, t, call=False)
        stock, rate = Stock(spot=s, dividend=q), Rate(rate=r)
        model = BSMModel(params={"sigma": sigma})
        for tech_name, tech in (
            ("CRR", CRRTechnique(steps=1000, is_american=True)),
            ("Leisen-Reimer", LeisenReimerTechnique(steps=501, is_american=True)),
            ("TOPM", TOPMTechnique(steps=1000, is_american=True)),
        ):
            got = _price(tech, option, stock, model, rate)
            _observe(
                rows,
                model="BSM American",
                technique=tech_name,
                reference=QL_AMERICAN,
                family="quantlib-fd",
                label=label,
                error=_err(got, ref),
            )
        amc = AmericanMonteCarloTechnique(
            n_paths=100_000, n_steps=50, seed=3, lsm_degree=4
        )
        got = _price(amc, option, stock, model, rate)
        _observe(
            rows,
            model="BSM American",
            technique="LSMC",
            reference=QL_AMERICAN,
            family="monte-carlo",
            label=label,
            error=_err(got, ref),
        )

    _log("characteristic-function techniques against each other")
    stock, rate = Stock(spot=100.0, dividend=0.01), Rate(rate=0.05)
    for name, (model, kwargs) in c["cf_models"]().items():
        for k, t in ((80.0, 0.25), (100.0, 1.0), (120.0, 2.0)):
            option = make_option(k, t)
            cos_price = _price(cos, option, stock, model, rate, **kwargs)
            for tech_name, tech in (("FFT", fft), ("Gauss-Legendre", quad)):
                got = _price(tech, option, stock, model, rate, **kwargs)
                _observe(
                    rows,
                    model=name,
                    technique=tech_name,
                    reference=COS_REF,
                    family="internal",
                    label=f"K={k:g} T={t:g}",
                    error=_err(got, cos_price),
                )
            if name == "BSM":
                analytic = _price(closed, option, stock, model, rate, **kwargs)
                _observe(
                    rows,
                    model="BSM",
                    technique="COS",
                    reference=BSM_CF,
                    family="internal",
                    label=f"K={k:g} T={t:g}",
                    error=_err(cos_price, analytic),
                )

    _log("convergence slopes")
    convergence = _convergence(c["errors"], c["slopes"])
    return rows, convergence


def _heston(v0, kappa, theta, vol_of_vol, rho):
    from optpricing import HestonModel

    return HestonModel(
        params={
            "v0": v0,
            "kappa": kappa,
            "theta": theta,
            "rho": rho,
            "vol_of_vol": vol_of_vol,
        }
    )


def _convergence(errors_fn, slopes_fn) -> list[dict]:
    import numpy as np

    from optpricing import (
        CRRTechnique,
        LeisenReimerTechnique,
        MonteCarloTechnique,
        PDETechnique,
        TOPMTechnique,
    )
    from tests.correctness._helpers import make_option
    from tests.correctness.test_convergence import MODEL, RATE, STOCK

    specs = [
        ("CRR", [200, 400, 800], lambda n: CRRTechnique(steps=n), -1.0),
        ("TOPM", [200, 400, 800], lambda n: TOPMTechnique(steps=n), -1.0),
        (
            "Leisen-Reimer",
            [101, 201, 401],
            lambda n: LeisenReimerTechnique(steps=n),
            -2.0,
        ),
        (
            "Crank-Nicolson",
            [100, 200, 400],
            lambda n: PDETechnique(M=n, N=n),
            -2.0,
        ),
    ]
    out = []
    for scheme, sizes, make, theory in specs:
        slopes = slopes_fn(errors_fn(make, sizes), np.asarray(sizes, dtype=float))
        values = [float(s) for s in slopes]
        out.append(
            {
                "scheme": scheme,
                "resolutions": sizes,
                "slopes": values,
                "mean_slope": float(np.mean(values)),
                "theory": theory,
            }
        )

    # Same seeds and sizes as the Monte Carlo slope test in test_convergence.py.
    from optpricing.techniques import ClosedFormTechnique

    option = make_option(100.0, 1.0)
    ref = ClosedFormTechnique().price(option, STOCK, MODEL, RATE).price
    spread = {}
    for n in (2048, 32768):
        prices = np.array(
            [
                MonteCarloTechnique(n_paths=n, n_steps=1, seed=k, antithetic=False)
                .price(option, STOCK, MODEL, RATE)
                .price
                for k in range(60)
            ]
        )
        spread[n] = float(prices.std(ddof=1))
        if abs(float(prices.mean()) - ref) >= 3.5 * spread[n] / math.sqrt(prices.size):
            _log(f"warning: MC mean at {n} paths is outside the test's bias band")
    slope = math.log(spread[32768] / spread[2048]) / math.log(32768 / 2048)
    out.append(
        {
            "scheme": "Monte Carlo",
            "resolutions": [2048, 32768],
            "slopes": [slope],
            "mean_slope": slope,
            "theory": -0.5,
        }
    )
    return out


def aggregate(rows: list[Obs]) -> list[dict]:
    order: list[tuple[str, str, str]] = []
    bucket: dict[tuple[str, str, str], dict] = {}
    for obs in rows:
        key = (obs.model, obs.technique, obs.reference)
        if key not in bucket:
            order.append(key)
            bucket[key] = {
                "model": obs.model,
                "technique": obs.technique,
                "reference": obs.reference,
                "family": obs.family,
                "n_cases": 0,
                "max_abs_error": 0.0,
                "worst_label": obs.label,
            }
        item = bucket[key]
        item["n_cases"] += 1
        if obs.error >= item["max_abs_error"]:
            item["max_abs_error"] = obs.error
            item["worst_label"] = obs.label
    return [bucket[key] for key in order]


def headlines(rows: list[dict]) -> dict[str, float]:
    def worst(pred) -> float:
        picked = [row["max_abs_error"] for row in rows if pred(row)]
        return max(picked) if picked else 0.0

    return {
        "quantlib_analytic_max_abs_error": worst(
            lambda row: row["family"] == "quantlib-analytic"
        ),
        "heston_quantlib_max_abs_error": worst(
            lambda row: row["model"] == "Heston" and row["reference"] == QL_HESTON
        ),
        "fft_vs_cos_max_abs_error": worst(
            lambda row: row["technique"] == "FFT" and row["reference"] == COS_REF
        ),
    }


def render_summary(rows: list[dict], *, worst: bool) -> str:
    headers = ["Model", "Technique", "Reference", "Cases", "Max abs error"]
    if worst:
        headers.append("Worst case")
    lines = [
        "| " + " | ".join(headers) + " |",
        "| --- | --- | --- | ---: | ---: |" + (" --- |" if worst else ""),
    ]
    for row in rows:
        cells = [
            row["model"],
            row["technique"],
            row["reference"],
            str(row["n_cases"]),
            format_error(row["max_abs_error"]),
        ]
        if worst:
            cells.append(str(row["worst_label"]).replace("|", "/"))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render_convergence(rows: list[dict]) -> str:
    lines = [
        "| Scheme | Resolutions | Measured slopes | Mean | Theory |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    for row in rows:
        if row["scheme"] == "Monte Carlo":
            resolutions = "2,048 and 32,768 paths"
        else:
            resolutions = ", ".join(str(n) for n in row["resolutions"])
        slopes = ", ".join(format_slope(s) for s in row["slopes"])
        lines.append(
            "| "
            + " | ".join(
                [
                    row["scheme"],
                    resolutions,
                    slopes,
                    format_slope(row["mean_slope"]),
                    format_slope(row["theory"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def payload_from(rows: list[Obs], convergence: list[dict]) -> dict:
    summary = aggregate(rows)
    return {
        "rows": summary,
        "convergence": convergence,
        "headlines": headlines(summary),
    }


def _span(name: str) -> tuple[str, str]:
    return f"<!-- {name} -->", f"<!-- /{name} -->"


def replace_span(text: str, name: str, body: str) -> str:
    start, end = _span(name)
    i = text.find(start)
    j = text.find(end)
    if i < 0 or j < 0 or j < i:
        raise SystemExit(f"{name} markers not found")
    return text[: i + len(start)] + "\n" + body.strip() + "\n" + text[j:]


def write_docs(payload: dict) -> None:
    summary = render_summary(payload["rows"], worst=False)
    full = render_summary(payload["rows"], worst=True)
    slopes = render_convergence(payload["convergence"])
    JSON_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    readme = README_PATH.read_text()
    readme = replace_span(readme, SUMMARY, summary)
    readme = replace_span(readme, CONVERGENCE, slopes)
    README_PATH.write_text(readme)
    guide = GUIDE_PATH.read_text()
    guide = replace_span(guide, SUMMARY_FULL, full)
    guide = replace_span(guide, CONVERGENCE, slopes)
    GUIDE_PATH.write_text(guide)


def _close(got: float, stored: float, tol: float) -> bool:
    scale = max(abs(got), abs(stored), 1e-18)
    return abs(got - stored) <= max(tol, 0.05 * scale)


def check(payload: dict) -> None:
    stored = json.loads(JSON_PATH.read_text())
    problems = []

    def index(rows):
        return {(r["model"], r["technique"], r["reference"]): r for r in rows}

    got_rows, stored_rows = index(payload["rows"]), index(stored["rows"])
    if got_rows.keys() != stored_rows.keys():
        missing = sorted(set(map(str, stored_rows)) - set(map(str, got_rows)))
        extra = sorted(set(map(str, got_rows)) - set(map(str, stored_rows)))
        problems.append(f"row keys differ, missing {missing}, extra {extra}")
    for key, stored_row in stored_rows.items():
        if key not in got_rows:
            continue
        got = got_rows[key]["max_abs_error"]
        old = stored_row["max_abs_error"]
        if not _close(got, old, 1e-15):
            problems.append(f"{key}: error {got:.3e} vs committed {old:.3e}")

    got_slopes = {r["scheme"]: r for r in payload["convergence"]}
    for row in stored["convergence"]:
        got = got_slopes.get(row["scheme"])
        if got is None:
            problems.append(f"missing slope {row['scheme']}")
            continue
        tol = 0.05 if row["scheme"] == "Monte Carlo" else 0.01
        if abs(got["mean_slope"] - row["mean_slope"]) > tol:
            problems.append(
                f"{row['scheme']} slope {got['mean_slope']:.4f} vs "
                f"{row['mean_slope']:.4f}"
            )

    readme = README_PATH.read_text()
    guide = GUIDE_PATH.read_text()
    expected = {
        (readme, SUMMARY): render_summary(stored["rows"], worst=False),
        (readme, CONVERGENCE): render_convergence(stored["convergence"]),
        (guide, SUMMARY_FULL): render_summary(stored["rows"], worst=True),
        (guide, CONVERGENCE): render_convergence(stored["convergence"]),
    }
    for (text, name), body in expected.items():
        start, end = _span(name)
        i, j = text.find(start), text.find(end)
        if i < 0 or j < 0:
            problems.append(f"missing {name} markers")
            continue
        found = text[i + len(start) : j].strip()
        if found != body.strip():
            problems.append(f"{name} markdown does not match the JSON")

    if problems:
        print("\n".join(problems), file=sys.stderr)
        raise SystemExit(1)
    print("accuracy check passed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--write", action="store_true", help="update JSON and docs")
    parser.add_argument(
        "--check", action="store_true", help="reprice and compare to the committed JSON"
    )
    args = parser.parse_args()
    warnings.filterwarnings(
        "ignore", message=".*Feller condition.*", category=UserWarning
    )
    rows, convergence = collect()
    payload = payload_from(rows, convergence)
    print(render_summary(payload["rows"], worst=True))
    print()
    print(render_convergence(payload["convergence"]))
    print()
    for name, value in payload["headlines"].items():
        print(f"{name}: {format_error(value)}")
    if args.write:
        write_docs(payload)
        print(f"\nwrote {JSON_PATH.relative_to(ROOT)}", file=sys.stderr)
    if args.check:
        check(payload)


if __name__ == "__main__":
    main()
