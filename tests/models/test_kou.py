import pytest

from optpricing.models import KouModel

# Common parameters for tests
PARAMS = {"sigma": 0.15, "lambda": 1.0, "p_up": 0.6, "eta1": 10.0, "eta2": 5.0}
CF_KWARGS = {"spot": 100, "r": 0.05, "q": 0.01, "t": 1.0}


@pytest.fixture
def model():
    """Provides a KouModel instance."""
    return KouModel(params=PARAMS)


def test_parameter_validation():
    """
    Tests that Kou rejects invalid parameters.
    """
    with pytest.raises(ValueError, match="parameters must be positive: eta1"):
        KouModel(params={**PARAMS, "eta1": -1.0})
    with pytest.raises(ValueError, match="parameter 'p_up' must be in"):
        KouModel(params={**PARAMS, "p_up": 1.1})


def test_characteristic_function(model):
    """
    Tests the characteristic function at u=0, where it should be 1.
    """
    cf = model.cf(**CF_KWARGS)
    assert cf(0) == pytest.approx(1.0)


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
