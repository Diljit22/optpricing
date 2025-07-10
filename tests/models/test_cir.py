import pytest

from optpricing.models import CIRModel

# Common parameters for tests
PARAMS = {"kappa": 0.86, "theta": 0.09, "sigma": 0.02}
PRICING_KWARGS = {"spot": 0.05, "t": 1.0}  # spot is r0, t is T


@pytest.fixture
def model():
    """Provides a CIRModel instance."""
    return CIRModel(params=PARAMS)


def test_parameter_validation():
    """
    Tests that CIR rejects invalid parameters.
    """
    with pytest.raises(ValueError, match="missing required parameters: sigma"):
        CIRModel(params={"kappa": 0.86, "theta": 0.09})
    with pytest.raises(ValueError, match="parameters must be positive: theta"):
        CIRModel(params={"kappa": 0.86, "theta": -0.09, "sigma": 0.02})


def test_closed_form_price(model):
    """
    Tests the closed-form bond price against a known 'golden' value.
    Value calculated from the analytic formula for the given parameters.
    """
    # For these params, the price should be higher than Vasicek due to lower
    # vol at low rates, making the bond less risky.
    expected_price = 0.9387
    price = model.price_closed_form(**PRICING_KWARGS)
    assert price == pytest.approx(expected_price, abs=1e-4)


def test_not_implemented_methods(model):
    """
    Tests that unsupported methods raise NotImplementedError.
    """
    with pytest.raises(NotImplementedError):
        model.cf()
    with pytest.raises(NotImplementedError):
        model.get_sde_sampler()
    with pytest.raises(NotImplementedError):
        model.get_pde_coeffs()
