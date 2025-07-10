from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from optpricing.atoms import Option, OptionType, Rate, Stock
from optpricing.techniques.base import BaseTechnique, IVMixin, PricingResult


class DummyIVTechnique(BaseTechnique, IVMixin):
    def __init__(self):
        # This mock will stand in for a real pricing engine like BSM's closed form
        self.bsm_pricer = MagicMock()

    def price(self, option, stock, model, rate, **kwargs):
        # The IVMixin calls `self.price` with a BSMModel instance.
        # We intercept this call and use our mock BSM pricer.
        return self.bsm_pricer(model.params["sigma"])


@pytest.fixture
def setup():
    """Provides a standard setup for IV tests."""
    technique = DummyIVTechnique()
    option = Option(strike=100, maturity=1.0, option_type=OptionType.CALL)
    stock = Stock(spot=100)
    model = MagicMock()
    rate = Rate(rate=0.05)
    return technique, option, stock, model, rate


def test_implied_volatility_brentq_success(setup):
    """
    Tests that the root finder can recover the correct volatility using brentq.
    """
    technique, option, stock, model, rate = setup
    target_vol = 0.25
    target_price = 10.0

    def mock_price_func(sigma):
        if abs(sigma - target_vol) < 1e-7:
            return PricingResult(price=target_price)
        elif sigma < target_vol:
            return PricingResult(price=target_price - 1)
        else:
            return PricingResult(price=target_price + 1)

    technique.bsm_pricer.side_effect = mock_price_func

    iv = technique.implied_volatility(
        option, stock, model, rate, target_price=target_price
    )
    assert iv == pytest.approx(target_vol, abs=1e-6)


@patch("optpricing.techniques.base.iv_mixin.brentq")
def test_implied_volatility_fallback_to_secant(mock_brentq, setup):
    """
    Tests that the secant method is called if brentq fails.
    """
    technique, option, stock, model, rate = setup
    target_price = 10.0

    # Make brentq fail, and mock the secant method to succeed
    mock_brentq.side_effect = ValueError("brentq failed")
    with patch.object(technique, "_secant_iv", return_value=0.3) as mock_secant:
        iv = technique.implied_volatility(
            option, stock, model, rate, target_price=target_price
        )

        mock_brentq.assert_called_once()
        mock_secant.assert_called_once()
        assert iv == 0.3


def test_implied_volatility_total_failure_returns_nan(setup):
    """
    Tests that nan is returned if both brentq and secant methods fail.
    """
    technique, option, stock, model, rate = setup
    target_price = 10.0

    # Make both methods fail
    with patch("optpricing.techniques.base.iv_mixin.brentq", side_effect=ValueError):
        with patch.object(technique, "_secant_iv", side_effect=RuntimeError):
            iv = technique.implied_volatility(
                option, stock, model, rate, target_price=target_price
            )
            assert np.isnan(iv)
