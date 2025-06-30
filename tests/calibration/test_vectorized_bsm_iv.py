import numpy as np
import pandas as pd
import pytest

from optpricing.atoms import Rate, Stock
from optpricing.calibration.vectorized_bsm_iv import BSMIVSolver
from optpricing.models import BSMModel


@pytest.fixture
def setup():
    """Provides a standard setup for BSM IV tests."""
    options = pd.DataFrame(
        {
            "strike": [95, 100, 105],
            "maturity": [1.0, 1.0, 1.0],
            "optionType": ["call", "call", "call"],
        }
    )
    stock = Stock(spot=100)
    rate = Rate(rate=0.05)
    return options, stock, rate


def test_bsm_iv_solver(setup):
    """
    Tests that the solver can recover known volatilities from BSM prices.
    """
    options, stock, rate = setup

    target_vols = np.array([0.15, 0.20, 0.25])
    target_prices = np.zeros_like(target_vols)
    for i, vol in enumerate(target_vols):
        model = BSMModel(params={"sigma": vol})
        target_prices[i] = model.price_closed_form(
            spot=stock.spot,
            strike=options.loc[i, "strike"],
            r=rate.get_rate(options.loc[i, "maturity"]),
            q=stock.dividend,
            t=options.loc[i, "maturity"],
            call=True,
        )

    solver = BSMIVSolver()
    implied_vols = solver.solve(target_prices, options, stock, rate)

    np.testing.assert_allclose(implied_vols, target_vols, atol=1e-3)
