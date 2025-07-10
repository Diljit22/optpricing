from unittest.mock import MagicMock, patch

import pytest

from optpricing.atoms import Option, OptionType, Rate, Stock, ZeroCouponBond
from optpricing.models import BSMModel, CIRModel, MertonJumpModel, VasicekModel
from optpricing.techniques import ClosedFormTechnique


# Common setup for tests
@pytest.fixture
def setup():
    option = Option(strike=100, maturity=1.0, option_type=OptionType.CALL)
    stock = Stock(spot=100)
    # Using BSMModel as it has both closed-form price and analytic greeks
    model = BSMModel(params={"sigma": 0.2})
    rate = Rate(rate=0.05)
    return option, stock, model, rate


def test_closed_form_model_type_check(setup):
    """
    Tests that the technique raises a TypeError if the model does not
    have a closed-form solution.
    """
    option, stock, _, rate = setup

    # Create a mock model that does not have a closed-form solution
    no_cf_model = MagicMock()
    no_cf_model.has_closed_form = False
    no_cf_model.name = "NoCFModel"

    technique = ClosedFormTechnique()
    with pytest.raises(TypeError, match="has no closed-form solver"):
        technique.price(option, stock, no_cf_model, rate)


def test_price_calls_model_closed_form(setup):
    """
    Tests that the price method correctly calls the model's implementation.
    """
    option, stock, model, rate = setup
    technique = ClosedFormTechnique()

    # Spy on the model's method to see if it gets called
    with patch.object(model, "price_closed_form", return_value=10.0) as mock_price:
        result = technique.price(option, stock, model, rate)

        mock_price.assert_called_once()
        assert result.price == 10.0


def test_analytic_greek_dispatch(setup):
    """
    Tests that the technique calls the model's analytic greek method when available.
    """
    option, stock, model, rate = setup
    technique = ClosedFormTechnique(use_analytic_greeks=True)

    # Spy on the model's analytic delta
    with patch.object(model, "delta_analytic", return_value=0.5) as mock_analytic_delta:
        delta = technique.delta(option, stock, model, rate)

        mock_analytic_delta.assert_called_once()
        assert delta == 0.5


def test_numerical_greek_fallback_when_disabled(setup):
    """
    Tests that the technique falls back to numerical greeks when analytic greeks
    are disabled.
    """
    option, stock, model, rate = setup
    technique = ClosedFormTechnique(use_analytic_greeks=False)

    # Spy on the superclass's (finite difference) delta method
    with patch.object(
        technique, "price", return_value=MagicMock(price=10.0)
    ) as mock_price:
        technique.delta(option, stock, model, rate)

        # The numerical delta calls price twice (for up and down shifts)
        assert mock_price.call_count == 2


def test_merton_closed_form_price(setup):
    """
    Tests the ClosedFormTechnique with the Merton Jump-Diffusion model against
    a known benchmark value.
    """
    option, stock, _, rate = setup

    merton_model = MertonJumpModel(
        params={
            "sigma": 0.2,
            "lambda": 0.5,
            "mu_j": -0.1,
            "sigma_j": 0.15,
            "max_sum_terms": 100,
        }
    )

    technique = ClosedFormTechnique()
    price_result = technique.price(option, stock, merton_model, rate)

    # Benchmark value calculated from a trusted external source
    expected_price = 11.6616
    assert price_result.price == pytest.approx(expected_price, rel=1e-4)


def test_vasicek_closed_form_price():
    """
    Tests the ClosedFormTechnique with the Vasicek interest rate model.
    """
    # For rate models, the 'stock' atom is used to hold the initial short rate r0
    r0_stock = Stock(spot=0.05)
    bond = ZeroCouponBond(maturity=1.0, face_value=1.0)
    dummy_rate = Rate(rate=0.0)  # The 'rate' atom is ignored for rate models

    vasicek_model = VasicekModel(
        params={
            "kappa": 0.86,
            "theta": 0.09,
            "sigma": 0.02,
        }
    )

    technique = ClosedFormTechnique()
    price_result = technique.price(bond, r0_stock, vasicek_model, dummy_rate)

    # Benchmark value from standard bond pricing formula for Vasicek
    expected_price = 0.93881
    assert price_result.price == pytest.approx(expected_price, rel=1e-3)


def test_cir_closed_form_price():
    """
    Tests the ClosedFormTechnique with the Cox-Ingersoll-Ross (CIR) rate model.
    """
    r0_stock = Stock(spot=0.05)
    bond = ZeroCouponBond(maturity=1.0, face_value=1.0)
    dummy_rate = Rate(rate=0.0)

    cir_model = CIRModel(
        params={
            "kappa": 0.86,
            "theta": 0.09,
            "sigma": 0.02,
        }
    )

    technique = ClosedFormTechnique()
    price_result = technique.price(bond, r0_stock, cir_model, dummy_rate)

    # Benchmark value from standard bond pricing formula for CIR
    expected_price = 0.93881
    assert price_result.price == pytest.approx(expected_price, rel=1e-4)


def test_unsupported_asset_type(setup):
    """
    Tests that the technique raises a TypeError for an unsupported asset type.
    """
    # Create a dummy object that is not an Option or ZeroCouponBond
    unsupported_asset = "a string"

    # Use any valid setup for the other parameters
    _, stock, model, rate = setup

    technique = ClosedFormTechnique()
    with pytest.raises(TypeError, match="Unsupported asset type"):
        technique.price(unsupported_asset, stock, model, rate)
