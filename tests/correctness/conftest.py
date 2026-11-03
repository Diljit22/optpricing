"""
Shared fixtures for the reference-value and invariant tests.

Reference values in this package were produced with QuantLib-Python 1.44
(evaluation date 2026-11-03, Actual/365 Fixed day count, maturities chosen as
whole numbers of days so the year fractions are exact) and with independent
textbook formulas, using a throwaway script that is intentionally not part of
the repository. Each test states the engine or formula it was checked against.
"""

from __future__ import annotations

import pytest

from optpricing import (
    BatesModel,
    BSMModel,
    CGMYModel,
    HestonModel,
    HyperbolicModel,
    KouModel,
    MertonJumpModel,
    NIGModel,
    Option,
    Rate,
    Stock,
    VarianceGammaModel,
)

from ._helpers import make_option


@pytest.fixture
def stock() -> Stock:
    return Stock(spot=100.0, dividend=0.01)


@pytest.fixture
def rate() -> Rate:
    return Rate(rate=0.05)


@pytest.fixture
def atm_call() -> Option:
    return make_option(100.0, 1.0, call=True)


@pytest.fixture
def atm_put() -> Option:
    return make_option(100.0, 1.0, call=False)


@pytest.fixture
def heston_base() -> HestonModel:
    return HestonModel(
        params={"v0": 0.04, "kappa": 2.0, "theta": 0.05, "rho": -0.7, "vol_of_vol": 0.5}
    )


@pytest.fixture
def cf_models() -> dict[str, tuple]:
    """Every model with a characteristic function, with its extra pricing kwargs."""
    return {
        "BSM": (BSMModel(params={"sigma": 0.2}), {}),
        "Heston": (
            HestonModel(
                params={
                    "v0": 0.04,
                    "kappa": 2.0,
                    "theta": 0.05,
                    "rho": -0.7,
                    "vol_of_vol": 0.5,
                }
            ),
            {"v0": 0.04},
        ),
        "Merton": (
            MertonJumpModel(
                params={
                    "sigma": 0.2,
                    "lambda": 0.5,
                    "mu_j": -0.1,
                    "sigma_j": 0.15,
                    "max_sum_terms": 100,
                }
            ),
            {},
        ),
        "Bates": (
            BatesModel(
                params={
                    "v0": 0.04,
                    "kappa": 2.0,
                    "theta": 0.05,
                    "rho": -0.7,
                    "vol_of_vol": 0.5,
                    "lambda": 0.5,
                    "mu_j": -0.1,
                    "sigma_j": 0.15,
                }
            ),
            {"v0": 0.04},
        ),
        "Kou": (
            KouModel(
                params={
                    "sigma": 0.15,
                    "lambda": 1.0,
                    "p_up": 0.6,
                    "eta1": 10.0,
                    "eta2": 5.0,
                }
            ),
            {},
        ),
        "VG": (
            VarianceGammaModel(params={"sigma": 0.2, "nu": 0.1, "theta": -0.14}),
            {},
        ),
        "NIG": (NIGModel(params={"alpha": 15.0, "beta": -5.0, "delta": 0.5}), {}),
        "CGMY": (CGMYModel(params={"C": 0.02, "G": 5.0, "M": 5.0, "Y": 1.2}), {}),
        "Hyperbolic": (
            HyperbolicModel(
                params={"alpha": 15.0, "beta": -5.0, "delta": 0.5, "mu": 0.0}
            ),
            {},
        ),
    }
