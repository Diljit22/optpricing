from unittest.mock import patch

import pytest

from optpricing.models.base import BaseModel


# A concrete implementation of BaseModel for testing purposes
class DummyModel(BaseModel):
    name = "Dummy"

    def _validate_params(self) -> None:
        pass

    def _cf_impl(self, **kwargs):
        pass

    def _sde_impl(self, **kwargs):
        pass

    def _pde_impl(self, **kwargs):
        pass

    def _closed_form_impl(self, *args, **kwargs):
        pass


@pytest.fixture
def dummy_model():
    """Provides a basic DummyModel instance."""
    return DummyModel(params={"p1": 0.1, "p2": 0.2})


def test_base_model_init_calls_validate(dummy_model):
    """
    Tests that the BaseModel's __init__ method calls _validate_params.
    """
    with patch.object(DummyModel, "_validate_params") as mock_validate:
        model = DummyModel(params={"p1": 0.1})
        mock_validate.assert_called_once()
        assert model.params == {"p1": 0.1}


def test_base_model_with_params(dummy_model):
    """
    Tests that with_params creates a new instance with updated parameters.
    """
    new_model = dummy_model.with_params(p1=0.5)

    assert isinstance(new_model, DummyModel)
    assert new_model is not dummy_model  # Should be a new object
    assert new_model.params == {"p1": 0.5, "p2": 0.2}
    assert dummy_model.params == {"p1": 0.1, "p2": 0.2}  # Original is unchanged


def test_base_model_repr(dummy_model):
    """
    Tests the string representation of the model.
    """
    assert repr(dummy_model) == "DummyModel(p1=0.1000, p2=0.2000)"


def test_base_model_eq_and_hash(dummy_model):
    """
    Tests equality and hashing logic.
    """
    model1 = DummyModel(params={"p1": 0.1, "p2": 0.2})
    model2 = DummyModel(params={"p1": 0.1, "p2": 0.2})
    model3 = DummyModel(params={"p1": 0.5, "p2": 0.2})

    assert model1 == model2
    assert model1 != model3
    assert model1 != "not a model"

    # Test hashing for use in sets/dicts
    model_set = {model1, model2, model3}
    assert len(model_set) == 2


@pytest.mark.parametrize(
    "method_name, flag_name, impl_name",
    [
        ("cf", "supports_cf", "_cf_impl"),
        ("get_sde_sampler", "supports_sde", "_sde_impl"),
        ("get_pde_coeffs", "supports_pde", "_pde_impl"),
        ("price_closed_form", "has_closed_form", "_closed_form_impl"),
    ],
)
def test_gatekeeper_methods(method_name, flag_name, impl_name):
    """
    Tests the gatekeeper methods that check a support flag before execution.
    """
    # Case 1: Flag is False, should raise NotImplementedError
    setattr(DummyModel, flag_name, False)
    model_unsupported = DummyModel(params={})
    with pytest.raises(NotImplementedError):
        getattr(model_unsupported, method_name)()

    # Case 2: Flag is True, should call the implementation
    setattr(DummyModel, flag_name, True)
    model_supported = DummyModel(params={})
    with patch.object(DummyModel, impl_name) as mock_impl:
        getattr(model_supported, method_name)()
        mock_impl.assert_called_once()
