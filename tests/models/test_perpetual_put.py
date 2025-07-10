import pytest

from optpricing.models import PerpetualPutModel

# Common parameters for tests
PARAMS = {"sigma": 0.20, "rate": 0.08}
PRICING_KWARGS = {"spot": 90, "strike": 100, "q": 0.03}


@pytest.fixture
def model():
    """Provides a PerpetualPutModel instance."""
    return PerpetualPutModel(params=PARAMS)


def test_parameter_validation():
    """
    Tests that the model validates its parameters correctly.
    """
    with pytest.raises(ValueError, match="missing required parameters: rate"):
        PerpetualPutModel(params={"sigma": 0.2})
    with pytest.raises(ValueError, match="parameters must be positive: sigma"):
        PerpetualPutModel(params={"sigma": -0.2, "rate": 0.05})


def test_price_when_holding_is_optimal(model):
    """
    Tests the closed-form price when the spot price is above the optimal
    exercise boundary (S > S*).
    """
    expected_price = 14.7796
    price = model.price_closed_form(**PRICING_KWARGS)
    assert price == pytest.approx(expected_price, abs=1e-3)


def test_price_when_exercise_is_optimal(model):
    """
    Tests the price when the spot price is at or below the optimal
    exercise boundary (S <= S*).
    """
    # For these params, S* is approx 71.42. Spot=70, so exercise.
    kwargs = {**PRICING_KWARGS, "spot": 70}
    expected_price = kwargs["strike"] - kwargs["spot"]  # K - S
    price = model.price_closed_form(**kwargs)
    assert price == pytest.approx(expected_price)


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
