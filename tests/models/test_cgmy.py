import numpy as np
import pytest

from optpricing.models import CGMYModel

# Common parameters for tests
PARAMS = {"C": 0.02, "G": 5.0, "M": 5.0, "Y": 1.2}
CF_KWARGS = {"spot": 100, "r": 0.05, "q": 0.01, "t": 1.0}
SAMPLER_KWARGS = {"T": 1.0, "size": 1000}


@pytest.fixture
def model():
    """Provides a CGMYModel instance."""
    return CGMYModel(params=PARAMS)


def test_parameter_validation():
    """
    Tests that CGMY rejects invalid parameters.
    """
    with pytest.raises(ValueError, match="parameters must be positive: C"):
        CGMYModel(params={"C": -0.02, "G": 5.0, "M": 5.0, "Y": 1.2})
    with pytest.raises(ValueError, match="Y must be less than 2"):
        CGMYModel(params={"C": 0.02, "G": 5.0, "M": 5.0, "Y": 2.5})


def test_characteristic_function(model):
    """
    Tests the characteristic function at u=0, where it should be 1.
    """
    cf = model.cf(**CF_KWARGS)
    assert cf(0) == pytest.approx(1.0)


def test_raw_characteristic_function(model):
    """
    Tests the raw characteristic function at u=0, where it should be 1.
    """
    raw_cf = model.raw_cf(t=CF_KWARGS["t"])
    assert raw_cf(0) == pytest.approx(1.0)


def test_sampler_not_implemented_for_y_not_1(model):
    """
    Tests that the sampler raises an error if Y is not 1.
    """
    rng = np.random.default_rng(0)
    with pytest.raises(NotImplementedError, match="only implemented for Y=1"):
        model.sample_terminal_log_return(**SAMPLER_KWARGS, rng=rng)


def test_sampler_for_y_is_1():
    """
    Smoke test for the sampler when Y=1.
    Ensures it runs and returns an array of the correct size.
    """
    y1_params = {"C": 0.02, "G": 5.0, "M": 5.0, "Y": 1.0}
    y1_model = CGMYModel(params=y1_params)
    rng = np.random.default_rng(0)

    samples = y1_model.sample_terminal_log_return(**SAMPLER_KWARGS, rng=rng)

    assert isinstance(samples, np.ndarray)
    assert len(samples) == SAMPLER_KWARGS["size"]


def test_sampler_reproducibility_for_y_is_1():
    """
    Tests that the sampler produces the same results for the same seed when Y=1.
    """
    y1_params = {"C": 0.02, "G": 5.0, "M": 5.0, "Y": 1.0}
    y1_model = CGMYModel(params=y1_params)

    rng1 = np.random.default_rng(42)
    samples1 = y1_model.sample_terminal_log_return(**SAMPLER_KWARGS, rng=rng1)

    rng2 = np.random.default_rng(42)
    samples2 = y1_model.sample_terminal_log_return(**SAMPLER_KWARGS, rng=rng2)

    np.testing.assert_array_equal(samples1, samples2)


def test_not_implemented_methods(model):
    """
    Tests that other unsupported methods raise NotImplementedError.
    """
    with pytest.raises(NotImplementedError):
        model.get_sde_sampler()
    with pytest.raises(NotImplementedError):
        model.price_closed_form()
    with pytest.raises(NotImplementedError):
        model.get_pde_coeffs()
