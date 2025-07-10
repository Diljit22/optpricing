import numpy as np
import pytest

from optpricing.models import CEVModel

# Common parameters for tests
PARAMS = {"sigma": 0.8, "gamma": 0.7}
SAMPLER_KWARGS = {"S0": 100, "r": 0.05, "T": 1.0, "size": 1000}


@pytest.fixture
def model():
    """Provides a CEVModel instance."""
    return CEVModel(params=PARAMS)


def test_parameter_validation():
    """
    Tests that CEV rejects invalid parameters.
    """
    with pytest.raises(ValueError, match="parameters must be positive: sigma"):
        CEVModel(params={"sigma": -0.8, "gamma": 0.7})


def test_sample_terminal_spot(model):
    """
    Smoke test for the exact sampler.
    Ensures it runs and returns an array of the correct size with positive values.
    """
    samples = model.sample_terminal_spot(**SAMPLER_KWARGS)

    assert isinstance(samples, np.ndarray)
    assert len(samples) == SAMPLER_KWARGS["size"]
    assert np.all(samples > 0)


def test_bsm_boundary_case():
    """
    Tests that when gamma is close to 1, the model behaves like BSM.
    """
    bsm_like_params = {"sigma": 0.2, "gamma": 1.0 - 1e-7}
    bsm_like_model = CEVModel(params=bsm_like_params)
    samples = bsm_like_model.sample_terminal_spot(**SAMPLER_KWARGS)

    # The mean of log(ST/S0) should be close to the BSM drift
    expected_mean_log_return = (
        SAMPLER_KWARGS["r"] - 0.5 * bsm_like_params["sigma"] ** 2
    ) * SAMPLER_KWARGS["T"]
    mean_log_return = np.mean(np.log(samples / SAMPLER_KWARGS["S0"]))

    # Looser tolerance for MC
    assert mean_log_return == pytest.approx(expected_mean_log_return, abs=0.1)
