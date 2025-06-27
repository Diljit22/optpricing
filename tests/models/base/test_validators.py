import pytest

from optpricing.models.base import ParamValidator

MODEL_NAME = "TestModel"


def test_validator_require_success():
    """
    Tests that 'require' passes when all keys are present.
    """
    params = {"a": 1, "b": 2}
    ParamValidator.require(params, ["a", "b"], model=MODEL_NAME)
    # No exception should be raised


def test_validator_require_failure():
    """
    Tests that 'require' raises ValueError for missing keys.
    """
    params = {"a": 1}
    with pytest.raises(ValueError, match="TestModel: missing required parameters: b"):
        ParamValidator.require(params, ["a", "b"], model=MODEL_NAME)


def test_validator_positive_success():
    """
    Tests that 'positive' passes for strictly positive values.
    """
    params = {"a": 1, "b": 0.001}
    ParamValidator.positive(params, ["a", "b"], model=MODEL_NAME)
    # No exception should be raised


@pytest.mark.parametrize("invalid_val", [0, -1])
def test_validator_positive_failure(invalid_val):
    """
    Tests that 'positive' raises ValueError for zero or negative values.
    """
    params = {"a": 1, "b": invalid_val}
    with pytest.raises(ValueError, match="TestModel: parameters must be positive: b"):
        ParamValidator.positive(params, ["a", "b"], model=MODEL_NAME)


def test_validator_bounded_success():
    """
    Tests that 'bounded' passes for values within the range.
    """
    params = {"rho": -0.5}
    ParamValidator.bounded(params, "rho", -1, 1, model=MODEL_NAME)
    # No exception should be raised


@pytest.mark.parametrize("invalid_val", [-1.1, 1.1])
def test_validator_bounded_failure(invalid_val):
    """
    Tests that 'bounded' raises ValueError for values outside the range.
    """
    params = {"rho": invalid_val}
    with pytest.raises(ValueError, match=r"must be in \[-1, 1\]"):
        ParamValidator.bounded(params, "rho", -1, 1, model=MODEL_NAME)
