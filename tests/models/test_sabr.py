import pytest

from optpricing.models import SABRModel

# Common parameters for tests
PARAMS = {"alpha": 0.5, "beta": 0.8, "rho": -0.6}


@pytest.fixture
def model():
    """Provides a SABRModel instance."""
    return SABRModel(params=PARAMS)


def test_parameter_validation():
    """
    Tests that SABR rejects invalid parameters.
    """
    with pytest.raises(ValueError, match="missing required parameters: beta"):
        SABRModel(params={"alpha": 0.5, "rho": -0.6})
    with pytest.raises(ValueError, match="parameters must be positive: alpha"):
        SABRModel(params={"alpha": -0.5, "beta": 0.8, "rho": -0.6})
    with pytest.raises(ValueError, match="parameter 'beta' must be in"):
        SABRModel(params={"alpha": 0.5, "beta": 1.1, "rho": -0.6})
    with pytest.raises(ValueError, match="parameter 'rho' must be in"):
        SABRModel(params={"alpha": 0.5, "beta": 0.8, "rho": -1.1})


def test_not_implemented_methods(model):
    """
    Tests that methods not applicable to SABR raise NotImplementedError.
    """
    with pytest.raises(NotImplementedError):
        model.get_sde_sampler()
    with pytest.raises(NotImplementedError):
        model.cf()
    with pytest.raises(NotImplementedError):
        model.price_closed_form()
    with pytest.raises(NotImplementedError):
        model.get_pde_coeffs()
