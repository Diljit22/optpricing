import numpy as np
import pytest

from optpricing.models import NIGModel

# Common parameters for tests
PARAMS = {"alpha": 15.0, "beta": -5.0, "delta": 0.5}
CF_KWARGS = {"spot": 100, "r": 0.05, "q": 0.01, "t": 1.0}
SAMPLER_KWARGS = {"T": 1.0, "size": 1000}


@pytest.fixture
def model():
    """Provides a NIGModel instance."""
    return NIGModel(params=PARAMS)


def test_parameter_validation():
    """
    Tests that NIG rejects invalid parameters.
    """
    with pytest.raises(ValueError, match="missing required parameters: delta"):
        NIGModel(params={"alpha": 15.0, "beta": -5.0})
    with pytest.raises(ValueError, match="parameters must be positive: alpha"):
        NIGModel(params={"alpha": -15.0, "beta": -5.0, "delta": 0.5})
    with pytest.raises(ValueError, match="must satisfy |beta| < alpha"):
        NIGModel(params={"alpha": 5.0, "beta": -15.0, "delta": 0.5})


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


def test_sample_terminal_log_return(model):
    """
    Smoke test for the exact sampler.
    Ensures it runs and returns an array of the correct size.
    """
    rng = np.random.default_rng(0)
    samples = model.sample_terminal_log_return(**SAMPLER_KWARGS, rng=rng)

    assert isinstance(samples, np.ndarray)
    assert len(samples) == SAMPLER_KWARGS["size"]


def test_sampler_reproducibility(model):
    """
    Tests that the sampler produces the same results for the same seed.
    """
    rng1 = np.random.default_rng(42)
    samples1 = model.sample_terminal_log_return(**SAMPLER_KWARGS, rng=rng1)

    rng2 = np.random.default_rng(42)
    samples2 = model.sample_terminal_log_return(**SAMPLER_KWARGS, rng=rng2)

    np.testing.assert_array_equal(samples1, samples2)


def test_not_implemented_methods(model):
    """
    Tests that unsupported methods raise NotImplementedError.
    """
    with pytest.raises(NotImplementedError):
        model.get_sde_sampler()
    with pytest.raises(NotImplementedError):
        model.price_closed_form()
    with pytest.raises(NotImplementedError):
        model.get_pde_coeffs()
