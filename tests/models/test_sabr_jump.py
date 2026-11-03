import pytest

from optpricing.models import SABRJumpModel

# Common parameters for tests
PARAMS = {
    "alpha": 0.5,
    "beta": 0.8,
    "rho": -0.6,
    "nu": 0.5,
    "lambda": 0.4,
    "mu_j": -0.1,
    "sigma_j": 0.15,
}


@pytest.fixture
def model():
    """Provides a SABRJumpModel instance."""
    return SABRJumpModel(params=PARAMS)


def test_parameter_validation():
    """
    Tests presence, positivity and range checks of the parameters.
    """
    with pytest.raises(ValueError, match="missing required parameters: lambda"):
        SABRJumpModel(params={"alpha": 0.5, "beta": 0.8, "rho": -0.6, "nu": 0.5})
    with pytest.raises(ValueError, match="parameters must be positive: sigma_j"):
        SABRJumpModel(params={**PARAMS, "sigma_j": 0.0})
    with pytest.raises(ValueError, match="'lambda' must be non-negative"):
        SABRJumpModel(params={**PARAMS, "lambda": -0.1})
    with pytest.raises(ValueError, match="parameter 'rho' must be in"):
        SABRJumpModel(params={**PARAMS, "rho": 1.0})


def test_legacy_three_parameter_form_is_mapped_to_nu_with_a_warning():
    legacy = {k: v for k, v in PARAMS.items() if k != "nu"}
    with pytest.warns(DeprecationWarning, match="Interpreting nu=alpha"):
        model = SABRJumpModel(params=legacy)
    assert model.params == PARAMS


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
