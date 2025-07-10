from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from optpricing.atoms import Option, OptionType, Rate, Stock
from optpricing.models import (
    BSMModel,
    CEVModel,
    HestonModel,
    MertonJumpModel,
    VarianceGammaModel,
)
from optpricing.techniques import ClosedFormTechnique, MonteCarloTechnique
from optpricing.techniques.kernels import mc_kernels


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


@patch("optpricing.techniques.monte_carlo.MonteCarloTechnique._simulate_sde_path")
def test_sde_path_dispatcher(mock_simulate_sde, setup):
    """
    Tests that the dispatcher correctly calls the SDE path simulator.
    """
    mock_simulate_sde.return_value = np.array([100.0])

    option, stock, rate = setup
    model = BSMModel()  # A standard SDE model
    technique = MonteCarloTechnique()

    technique.price(option, stock, model, rate)
    mock_simulate_sde.assert_called_once()


@patch("optpricing.techniques.monte_carlo.MonteCarloTechnique._simulate_levy_terminal")
def test_levy_terminal_dispatcher(mock_simulate_levy, setup):
    """
    Tests that the dispatcher correctly calls the pure Levy terminal simulator.
    """
    mock_simulate_levy.return_value = np.array([100.0])

    option, stock, rate = setup
    model = VarianceGammaModel()  # A pure Levy model
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

    with patch.object(
        model, "sample_terminal_spot", return_value=np.array([100.0])
    ) as mock_exact_sampler:
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


def test_mc_price_correctness_for_bsm(setup):
    """
    Tests that the Monte Carlo price for a BSM model is statistically close
    to the analytical closed-form price.
    """
    option, stock, rate = setup
    model = BSMModel(params={"sigma": 0.2})

    # Calculate the analytical price to use as a benchmark
    analytic_technique = ClosedFormTechnique()
    expected_price = analytic_technique.price(option, stock, model, rate).price

    mc_technique = MonteCarloTechnique(n_paths=20000, n_steps=10, seed=0)
    mc_price = mc_technique.price(option, stock, model, rate).price

    assert mc_price == pytest.approx(expected_price, abs=1e-2)
