import pytest

from optpricing.models import DupireLocalVolModel


# A sample volatility surface function for testing
def sample_surface(T, K):
    return 0.2 + 0.1 * (K / 100 - 1) - 0.05 * T


@pytest.fixture
def model():
    """Provides a DupireLocalVolModel instance with a sample surface."""
    return DupireLocalVolModel(params={"vol_surface": sample_surface})


def test_model_initialization(model):
    """
    Tests that the model correctly stores the callable parameter.
    """
    assert callable(model.params["vol_surface"])
    assert model.params["vol_surface"] == sample_surface


def test_model_representation(model):
    """
    Tests that the __repr__ method correctly identifies the function name.
    """
    assert repr(model) == "DupireLocalVolModel(vol_surface=sample_surface)"


def test_model_equality(model):
    """
    Tests the equality comparison between models.
    """
    model1 = DupireLocalVolModel(params={"vol_surface": sample_surface})
    model2 = DupireLocalVolModel(params={"vol_surface": lambda T, K: 0.2})

    assert model == model1
    assert model != model2
    assert model != "not a model"


def test_not_implemented_methods(model):
    """
    Tests that standard methods not applicable to Dupire raise NotImplementedError.
    """
    with pytest.raises(NotImplementedError):
        model.get_sde_sampler()
    with pytest.raises(NotImplementedError):
        model.cf()
    with pytest.raises(NotImplementedError):
        model.price_closed_form()
    with pytest.raises(NotImplementedError):
        model.get_pde_coeffs()
