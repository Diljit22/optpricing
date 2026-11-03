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
    with pytest.raises(ValueError, match="M must exceed 1"):
        CGMYModel(params={"C": 0.02, "G": 5.0, "M": 0.9, "Y": 1.2})
    with pytest.raises(ValueError, match="Y must be non-negative"):
        CGMYModel(params={"C": 0.02, "G": 5.0, "M": 5.0, "Y": -0.5})


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


@pytest.mark.parametrize("y", [0.0, 0.5, 1.0, 1.2])
def test_sampler_runs_for_all_y(y):
    """
    The terminal sampler is available for every valid Y: exact gamma-difference
    sampling at Y=0 and characteristic-function inversion otherwise.
    """
    cgmy = CGMYModel(params={"C": 0.02, "G": 5.0, "M": 5.0, "Y": y})
    rng = np.random.default_rng(0)

    samples = cgmy.sample_terminal_log_return(**SAMPLER_KWARGS, rng=rng)

    assert isinstance(samples, np.ndarray)
    assert len(samples) == SAMPLER_KWARGS["size"]
    assert np.all(np.isfinite(samples))


@pytest.mark.parametrize("y", [0.0, 1.2])
def test_sampler_reproducibility(y):
    """
    Tests that the sampler produces the same results for the same seed.
    """
    cgmy = CGMYModel(params={"C": 0.02, "G": 5.0, "M": 5.0, "Y": y})

    rng1 = np.random.default_rng(42)
    samples1 = cgmy.sample_terminal_log_return(**SAMPLER_KWARGS, rng=rng1)

    rng2 = np.random.default_rng(42)
    samples2 = cgmy.sample_terminal_log_return(**SAMPLER_KWARGS, rng=rng2)

    np.testing.assert_array_equal(samples1, samples2)


@pytest.mark.parametrize("y_limit, y_near", [(0.0, 1e-9), (1.0, 1.0 + 1e-9)])
def test_cf_is_continuous_at_gamma_poles(y_limit, y_near):
    """
    Gamma(-Y) has poles at Y=0 and Y=1; the Levy exponent is finite there and the
    closed-form limits must agree with the general formula nearby.
    """
    u = np.array([0.3, 1.5, 6.0])
    at_limit = CGMYModel(params={"C": 0.3, "G": 6.0, "M": 9.0, "Y": y_limit})
    near_limit = CGMYModel(params={"C": 0.3, "G": 6.0, "M": 9.0, "Y": y_near})
    # the general formula is used for y_near (np.isclose default atol is 1e-8)
    near_limit_params = dict(near_limit.params, Y=y_near + 2e-8)
    near_limit = CGMYModel(params=near_limit_params)

    np.testing.assert_allclose(
        at_limit.raw_cf(t=1.0)(u), near_limit.raw_cf(t=1.0)(u), rtol=1e-6
    )


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
