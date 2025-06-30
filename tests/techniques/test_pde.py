from unittest.mock import MagicMock

import pytest

from optpricing.atoms import Option, OptionType, Rate, Stock
from optpricing.models import BSMModel
from optpricing.techniques import PDETechnique


# Common setup for tests
@pytest.fixture
def setup():
    option = Option(strike=105, maturity=1.0, option_type=OptionType.CALL)
    stock = Stock(spot=100)
    model = BSMModel(params={"sigma": 0.2})
    rate = Rate(rate=0.05)
    return option, stock, model, rate


def test_pde_model_type_check(setup):
    """
    Tests that the PDETechnique raises a TypeError if not used with BSMModel.
    """
    option, stock, _, rate = setup

    # Create a mock model that is not a BSMModel instance
    non_bsm_model = MagicMock()
    non_bsm_model.name = "NotBSM"

    technique = PDETechnique()
    with pytest.raises(TypeError, match="optimized for BSMModel only"):
        technique.price(option, stock, non_bsm_model, rate)


def test_pde_price_convergence(setup):
    """
    Tests that the PDE price converges to the BSM analytical price.
    """
    option, stock, model, rate = setup

    # Get the analytical BSM price as the benchmark
    bsm_price = model.price_closed_form(
        spot=stock.spot,
        strike=option.strike,
        r=rate.get_rate(option.maturity),
        q=stock.dividend,
        t=option.maturity,
        call=(option.option_type is OptionType.CALL),
    )

    # Use a reasonably fine grid for the test
    technique = PDETechnique(M=500, N=500)
    pde_price = technique.price(option, stock, model, rate).price

    assert pde_price == pytest.approx(bsm_price, abs=1e-2)


def test_pde_greeks_accuracy(setup):
    """
    Tests that the grid-based Greeks are close to the analytical BSM Greeks.
    """
    option, stock, model, rate = setup

    # Get analytical Greeks as the benchmark
    bsm_delta = model.delta_analytic(
        spot=stock.spot,
        strike=option.strike,
        r=rate.get_rate(option.maturity),
        q=stock.dividend,
        t=option.maturity,
        call=(option.option_type is OptionType.CALL),
    )
    bsm_gamma = model.gamma_analytic(
        spot=stock.spot,
        strike=option.strike,
        r=rate.get_rate(option.maturity),
        q=stock.dividend,
        t=option.maturity,
    )

    # Use a reasonably fine grid for the test
    technique = PDETechnique(M=500, N=500)

    # The price() call populates the cache for Greeks
    technique.price(option, stock, model, rate)

    pde_delta = technique.delta(option, stock, model, rate)
    pde_gamma = technique.gamma(option, stock, model, rate)

    assert pde_delta == pytest.approx(bsm_delta, abs=1e-2)
    assert pde_gamma == pytest.approx(bsm_gamma, abs=1e-2)
