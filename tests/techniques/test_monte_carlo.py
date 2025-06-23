from unittest.mock import MagicMock, patch

import pytest

from quantfin.atoms import Option, OptionType, Rate, Stock
from quantfin.models import (
    BSMModel,
    CEVModel,
    HestonModel,
    MertonJumpModel,
    VarianceGammaModel,
)
from quantfin.techniques import MonteCarloTechnique
from quantfin.techniques.kernels import mc_kernels


# Common setup for tests
@pytest.fixture
def setup():
    option = Option(strike=100, maturity=1.0, option_type=OptionType.CALL)
    stock = Stock(spot=100)
    rate = Rate(rate=0.05)
    return option, stock, rate


def test_mc_model_support_check(setup):
    """
    Tests the tech raises a TypeError for models that don't support simulation.
    """
    option, stock, rate = setup

    # Create a mock model that supports nothing
    unsupported_model = MagicMock()
    unsupported_model.supports_sde = False
    unsupported_model.is_pure_levy = False
    unsupported_model.name = "Unsupported"

    technique = MonteCarloTechnique()
    with pytest.raises(TypeError, match="does not support simulation"):
        technique.price(option, stock, unsupported_model, rate)


@patch("quantfin.techniques.monte_carlo.MonteCarloTechnique._simulate_sde_path")
def test_sde_path_dispatcher(mock_simulate_sde, setup):
    """
    Tests that the dispatcher correctly calls the SDE path simulator.
    """
    option, stock, rate = setup
    model = BSMModel()  # A standard SDE model
    technique = MonteCarloTechnique()

    technique.price(option, stock, model, rate)
    mock_simulate_sde.assert_called_once()


@patch("quantfin.techniques.monte_carlo.MonteCarloTechnique._simulate_levy_terminal")
def test_levy_terminal_dispatcher(mock_simulate_levy, setup):
    """
    Tests that the dispatcher correctly calls the pure Lévy terminal simulator.
    """
    option, stock, rate = setup
    model = VarianceGammaModel()  # A pure Lévy model
    technique = MonteCarloTechnique()

    technique.price(option, stock, model, rate)
    mock_simulate_levy.assert_called_once()


def test_exact_sampler_dispatcher(setup):
    """
    Tests that the dispatcher correctly calls the model's exact sampler.
    """
    option, stock, rate = setup
    model = CEVModel()  # A model with an exact sampler
    technique = MonteCarloTechnique()

    # Spy on the model's method
    with patch.object(model, "sample_terminal_spot") as mock_exact_sampler:
        technique.price(option, stock, model, rate)
        mock_exact_sampler.assert_called_once()


@pytest.mark.parametrize(
    "model_instance, expected_kernel",
    [
        (BSMModel(), mc_kernels.bsm_kernel),
        (HestonModel(), mc_kernels.heston_kernel),
        (
            MertonJumpModel(params=MertonJumpModel.default_params),
            mc_kernels.merton_kernel,
        ),
    ],
)
def test_get_sde_kernel_selector(model_instance, expected_kernel):
    """
    Tests that the kernel selector returns the correct kernel for each model type.
    """
    technique = MonteCarloTechnique()
    kernel_func, _ = technique._get_sde_kernel_and_params(
        model=model_instance, r=0.05, q=0.01, dt=0.01
    )
    assert kernel_func is expected_kernel
