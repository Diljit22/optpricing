import numpy as np
import pytest

from optpricing.models import BlacksApproxModel

# Common parameters for tests
PARAMS = {"sigma": 0.30}
PRICING_KWARGS = {"spot": 100, "strike": 100, "r": 0.05, "t": 0.5}


@pytest.fixture
def model():
    """Provides a BlacksApproxModel instance."""
    return BlacksApproxModel(params=PARAMS)


def test_parameter_validation(model):
    """
    Tests that the model validates the presence and sign of sigma.
    """
    with pytest.raises(ValueError, match="missing required parameters: sigma"):
        BlacksApproxModel(params={})
    with pytest.raises(ValueError, match="parameters must be positive: sigma"):
        BlacksApproxModel(params={"sigma": -0.1})


def test_value_error_for_no_dividends(model):
    """
    Tests that the model raises an error if no dividends are provided.
    """
    with pytest.raises(ValueError, match="requires non-empty 'discrete_dividends'"):
        model.price_closed_form(
            **PRICING_KWARGS, discrete_dividends=np.array([]), ex_div_times=np.array([])
        )


def test_not_implemented_for_puts(model):
    """
    Tests that the model raises an error for put options.
    """
    with pytest.raises(NotImplementedError, match="for American calls only"):
        model.price_closed_form(
            **PRICING_KWARGS,
            call=False,
            discrete_dividends=np.array([1]),
            ex_div_times=np.array([0.1]),
        )


def test_price_when_holding_is_optimal(model):
    """
    Tests the price when the dividend is small, making holding to maturity optimal.
    """
    # Small dividend, so holding should be optimal.
    divs = np.array([0.1])
    div_times = np.array([0.25])

    # Expected price is BSM on spot adjusted for PV of dividends
    pv_div = divs[0] * np.exp(-PRICING_KWARGS["r"] * div_times[0])
    adj_spot = PRICING_KWARGS["spot"] - pv_div
    expected_price = model.bsm_solver.price_closed_form(
        spot=adj_spot,
        strike=PRICING_KWARGS["strike"],
        r=PRICING_KWARGS["r"],
        q=0,
        t=PRICING_KWARGS["t"],
        call=True,
    )

    price = model.price_closed_form(
        **PRICING_KWARGS,
        discrete_dividends=divs,
        ex_div_times=div_times,
    )
    assert price == pytest.approx(expected_price)


def test_price_when_early_exercise_is_optimal(model):
    """
    Tests the price when a large dividend makes early exercise optimal.
    """
    # Large dividend, making early exercise just before t=0.25 optimal
    divs = np.array([10.0])
    div_times = np.array([0.25])

    # Expected price is the BSM price at the ex-dividend date
    expected_price = model.bsm_solver.price_closed_form(
        spot=PRICING_KWARGS["spot"],
        strike=PRICING_KWARGS["strike"],
        r=PRICING_KWARGS["r"],
        q=0,
        t=div_times[0],
        call=True,
    )

    price = model.price_closed_form(
        **PRICING_KWARGS, discrete_dividends=divs, ex_div_times=div_times
    )
    assert price == pytest.approx(expected_price)
