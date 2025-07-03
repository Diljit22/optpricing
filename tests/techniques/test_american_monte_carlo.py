from __future__ import annotations

import pytest

from optpricing.atoms import Option, OptionType, Rate, Stock
from optpricing.models import BSMModel
from optpricing.techniques import AmericanMonteCarloTechnique, CRRTechnique


@pytest.fixture
def setup():
    """Provides common objects for American option tests."""
    # Use parameters known to have significant early exercise premium
    option = Option(strike=110.0, maturity=1.0, option_type=OptionType.PUT)
    stock = Stock(spot=100.0, dividend=0.01)
    rate = Rate(rate=0.05)
    return option, stock, rate


def test_american_mc_bsm_price_against_lattice(setup):
    """
    Tests the American MC price for BSM against a CRR lattice benchmark.
    This is the most important validation test.
    """
    option, stock, rate = setup
    model = BSMModel(params={"sigma": 0.2})

    # Calculate benchmark price using a high-resolution binomial tree
    lattice_pricer = CRRTechnique(steps=1001, is_american=True)
    benchmark_price = lattice_pricer.price(option, stock, model, rate).price

    american_mc_pricer = AmericanMonteCarloTechnique(
        n_paths=50000,
        n_steps=100,
        seed=42,
    )
    mc_price = american_mc_pricer.price(option, stock, model, rate).price

    assert mc_price == pytest.approx(benchmark_price, abs=0.1)


def test_get_path_kernel_selector():
    """
    Tests that the internal kernel dispatcher selects the correct path kernel.
    """
    from optpricing.models import HestonModel
    from optpricing.techniques.kernels import path_kernels

    technique = AmericanMonteCarloTechnique()
    heston_model = HestonModel()

    kernel_func, _ = technique._get_path_kernel_and_params(
        model=heston_model, r=0.05, q=0.01, dt=0.01, v0=0.04
    )

    assert kernel_func is path_kernels.heston_path_kernel
