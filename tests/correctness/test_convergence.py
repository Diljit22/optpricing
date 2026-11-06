"""
Convergence orders of the discretised techniques against the BSM closed form
(S=100, sigma 25 %, r=5 %, q=1 %, T=1). Empirical slopes of log|error| vs log N
must match the theory of each scheme, so a lost order (e.g. Rannacher start-up
or the Peizer-Pratt inversion silently dropped) fails even when the price at the
default resolution is still "close".
"""

from __future__ import annotations

import numpy as np
import pytest

from optpricing import (
    BSMModel,
    CRRTechnique,
    IntegrationTechnique,
    LeisenReimerTechnique,
    MonteCarloTechnique,
    PDETechnique,
    Rate,
    Stock,
    TOPMTechnique,
)
from optpricing.techniques import ClosedFormTechnique

from ._helpers import make_option

MODEL = BSMModel(params={"sigma": 0.25})
STOCK, RATE = Stock(spot=100.0, dividend=0.01), Rate(rate=0.05)


def _errors(make_technique, sizes, K=100.0):
    option = make_option(K, 1.0)
    ref = ClosedFormTechnique().price(option, STOCK, MODEL, RATE).price
    return np.array(
        [
            abs(make_technique(n).price(option, STOCK, MODEL, RATE).price - ref)
            for n in sizes
        ]
    )


def _slopes(errors, sizes):
    return np.diff(np.log(errors)) / np.diff(np.log(sizes))


@pytest.mark.parametrize("technique", [CRRTechnique, TOPMTechnique])
def test_lattices_converge_at_first_order(technique):
    """At the money the CRR/Kamrad-Ritchken error is c/N (Leisen & Reimer 1996)."""
    sizes = np.array([200, 400, 800])
    slopes = _slopes(_errors(lambda n: technique(steps=n), sizes), sizes)
    assert np.all((slopes > -1.1) & (slopes < -0.9))


@pytest.mark.parametrize("K", [100.0, 95.0])
def test_leisen_reimer_converges_at_second_order_for_odd_steps(K):
    """Leisen & Reimer (1996): smooth O(N^-2) convergence for odd N, any strike."""
    sizes = np.array([101, 201, 401])
    slopes = _slopes(_errors(lambda n: LeisenReimerTechnique(steps=n), sizes, K), sizes)
    assert np.all((slopes > -2.1) & (slopes < -1.9))


@pytest.mark.parametrize("K", [100.0, 95.0])
def test_crank_nicolson_converges_at_second_order(K):
    """Crank-Nicolson with Rannacher start-up: O(dS^2 + dt^2) (Rannacher 1984)."""
    sizes = np.array([100, 200, 400])
    slopes = _slopes(_errors(lambda n: PDETechnique(M=n, N=n), sizes, K), sizes)
    assert np.all((slopes > -2.2) & (slopes < -1.8))


def test_gauss_legendre_converges_exponentially_in_nodes():
    """Gil-Pelaez with n Gauss-Legendre nodes per panel: error ~ rho^(-2n)."""
    errors = _errors(lambda n: IntegrationTechnique(n_nodes=n), [4, 6, 8])
    assert errors[0] / errors[1] > 50 and errors[1] / errors[2] > 50
    assert errors[2] < 1e-8


def test_monte_carlo_standard_error_scales_as_inverse_sqrt_paths():
    """16x the paths must shrink the seed-to-seed spread 4x, with no bias."""
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
        spread[n] = prices.std(ddof=1)
        assert abs(prices.mean() - ref) < 3.5 * spread[n] / np.sqrt(prices.size)
    assert 3.0 < spread[2048] / spread[32768] < 5.3
