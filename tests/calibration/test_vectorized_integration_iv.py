import numpy as np
import pandas as pd
import pytest

from optpricing.atoms import Rate
from optpricing.calibration.vectorized_integration_iv import (
    VectorizedIntegrationIVSolver,
)
from optpricing.models import BSMModel


@pytest.fixture
def setup():
    """Provides a standard setup for integration IV tests."""
    options = pd.DataFrame(
        {
            "strike": [95, 100, 105],
            "maturity": [1.0, 1.0, 1.0],
            "optionType": ["call", "call", "call"],
            "spot": [100, 100, 100],
            "dividend": [0.0, 0.0, 0.0],
        }
    )
    rate = Rate(rate=0.05)
    return options, rate


def test_vectorized_integration_iv_solver(setup):
    """
    Tests that the solver can recover known volatilities from BSM prices
    calculated via integration.
    """
    options, rate = setup

    target_vols = np.array([0.15, 0.20, 0.25])
    target_prices = np.zeros_like(target_vols)

    # The solver's internal pricer is BSM via integration, so we use that
    # to generate the target prices for a fair comparison.
    solver = VectorizedIntegrationIVSolver()
    target_prices = solver._price_vectorized(target_vols, options, BSMModel(), rate)

    implied_vols = solver.solve(target_prices, options, BSMModel(), rate)

    np.testing.assert_allclose(implied_vols, target_vols, atol=1e-3)
