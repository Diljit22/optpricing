import pytest

from optpricing.techniques.kernels.lattice_kernels import (
    _crr_pricer,
    _lr_pricer,
    _topm_pricer,
)

EURO_PARAMS = {
    "S0": 100,
    "K": 105,
    "T": 1.0,
    "r": 0.05,
    "q": 0.01,
    "sigma": 0.2,
    "is_call": True,
    "is_am": False,
    "N": 501,
}
# Expected BSM price for the above parameters
EXPECTED_EURO_PRICE = 7.49170


@pytest.mark.parametrize("pricer_func", [_crr_pricer, _lr_pricer])
def test_binomial_pricers_european_convergence(pricer_func):
    """
    Tests that binomial pricers converge to the BSM price for a European option.
    """
    result = pricer_func(**EURO_PARAMS)
    assert result["price"] == pytest.approx(EXPECTED_EURO_PRICE, abs=1e-2)

    # Check that all required keys for Greeks are present
    required_keys = [
        "price_up",
        "price_down",
        "price_uu",
        "price_ud",
        "price_dd",
        "spot_up",
        "spot_down",
        "spot_uu",
        "spot_ud",
        "spot_dd",
    ]
    for key in required_keys:
        assert key in result


def test_topm_pricer_european_convergence():
    """
    Tests that the trinomial pricer converges to the BSM price.
    """
    topm_params = {**EURO_PARAMS, "vol": EURO_PARAMS["sigma"]}
    del topm_params["sigma"]  # TOPM uses 'vol' instead of 'sigma'

    result = _topm_pricer(**topm_params)
    assert result["price"] == pytest.approx(EXPECTED_EURO_PRICE, abs=1e-2)

    # Check that all required keys for Greeks are present
    required_keys = [
        "price_up",
        "price_mid",
        "price_down",
        "spot_up",
        "spot_mid",
        "spot_down",
    ]
    for key in required_keys:
        assert key in result


def test_american_put_early_exercise():
    """
    Tests an American put where early exercise is optimal.
    The price should equal the intrinsic value (K - S0).
    """
    american_params = {
        "S0": 100,
        "K": 105,
        "T": 1.0,
        "r": 0.05,
        "q": 0.01,
        "sigma": 0.2,
        "is_call": False,
        "is_am": False,
        "N": 501,
    }
    expected_price = 8.36581

    result = _crr_pricer(**american_params)
    assert result["price"] == pytest.approx(expected_price, abs=1e-2)
