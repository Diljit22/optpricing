import pytest

from optpricing.models import SABRJumpModel

# Common parameters for tests
PARAMS = {
    "alpha": 0.5,
    "beta": 0.8,
    "rho": -0.6,
    "lambda": 0.4,
    "mu_j": -0.1,
    "sigma_j": 0.15,
}


@pytest.fixture
def model():
    """Provides a SABRJumpModel instance."""
    return SABRJumpModel(params=PARAMS)


def test_parameter_validation(model):
    """
    Tests that the model validates the presence of all required parameters.
    Note: More detailed validation (positive, bounded) would be added here
    if implemented in the source code.
    """
    with pytest.raises(ValueError, match="missing required parameters: lambda"):
        SABRJumpModel(params={"alpha": 0.5, "beta": 0.8, "rho": -0.6})


def test_not_implemented_methods(model):
    """
    Tests that methods not applicable to SABRJump raise NotImplementedError.
    """
    with pytest.raises(NotImplementedError):
        model.get_sde_sampler()
    with pytest.raises(NotImplementedError):
        model.cf()
    with pytest.raises(NotImplementedError):
        model.price_closed_form()
    with pytest.raises(NotImplementedError):
        model.get_pde_coeffs()
