from optpricing.techniques.base import PricingResult


def test_pricing_result_creation():
    """
    Tests that the PricingResult dataclass correctly stores all attributes.
    """
    greeks = {"delta": 0.5}
    result = PricingResult(price=10.5, greeks=greeks, implied_vol=0.2)
    assert result.price == 10.5
    assert result.greeks["delta"] == 0.5
    assert result.implied_vol == 0.2


def test_pricing_result_defaults():
    """
    Tests that the default factory for greeks works correctly.
    """
    result1 = PricingResult(price=10.0)
    result2 = PricingResult(price=11.0)
    assert result1.greeks == {}
    assert result1.implied_vol is None

    # Test that default dicts are independent instances
    result1.greeks["delta"] = 0.5
    assert "delta" not in result2.greeks
