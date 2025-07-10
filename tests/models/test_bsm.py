import numpy as np
import pytest

from optpricing.models import BSMModel

# Common parameters for tests
PARAMS = {"sigma": 0.2}
PRICING_KWARGS = {"spot": 100, "strike": 105, "r": 0.05, "q": 0.01, "t": 1.0}


@pytest.fixture
def model():
    """Provides a BSMModel instance."""
    return BSMModel(params=PARAMS)


def test_parameter_validation():
    """
    Tests that BSM rejects invalid parameters.
    """
    with pytest.raises(ValueError, match="parameters must be positive: sigma"):
        BSMModel(params={"sigma": -0.1})


def test_closed_form_price(model):
    """
    Tests the closed-form price against a known 'golden' value.
    Value from: https://www.math.drexel.edu/~pg/fin/VanillaCalculator.html
    """
    call_price = model.price_closed_form(**PRICING_KWARGS, call=True)
    put_price = model.price_closed_form(**PRICING_KWARGS, call=False)
    assert call_price == pytest.approx(7.492, abs=1e-3)
    assert put_price == pytest.approx(8.366, abs=1e-3)


def test_put_call_parity(model):
    """
    Tests that the closed-form prices satisfy put-call parity.
    """
    call_price = model.price_closed_form(**PRICING_KWARGS, call=True)
    put_price = model.price_closed_form(**PRICING_KWARGS, call=False)
    S, K, r, q, T = PRICING_KWARGS.values()

    parity_lhs = call_price - put_price
    parity_rhs = S * np.exp(-q * T) - K * np.exp(-r * T)
    assert parity_lhs == pytest.approx(parity_rhs)


def test_analytic_greeks(model):
    """
    Tests analytic greeks against known 'golden' values.
    Values from: https://www.math.drexel.edu/~pg/fin/VanillaCalculator.html
    """
    assert model.delta_analytic(**PRICING_KWARGS, call=True) == (
        pytest.approx(0.51715, abs=1e-3)
    )
    assert model.gamma_analytic(**PRICING_KWARGS) == pytest.approx(0.019717, abs=1e-4)
    assert model.vega_analytic(**PRICING_KWARGS) == pytest.approx(39.43528, abs=1e-3)
    assert model.theta_analytic(**PRICING_KWARGS, call=True) == (
        pytest.approx(-5.6375, abs=1e-4)
    )
    assert model.rho_analytic(**PRICING_KWARGS, call=True) == (
        pytest.approx(44.22343, abs=1e-3)
    )


def test_characteristic_function(model):
    """
    Tests the characteristic function at u=0, where it should be 1.
    """
    cf = model.cf(**PRICING_KWARGS)
    assert cf(0) == pytest.approx(1.0)


def test_sde_stepper(model):
    """
    Tests the SDE stepper for a single step with no randomness.
    """
    stepper = model.get_sde_sampler()
    log_s0 = np.log(PRICING_KWARGS["spot"])
    r, q, dt = 0.05, 0.01, 0.01
    dw = np.array([0.0])  # No random shock

    expected_log_s1 = log_s0 + (r - q - 0.5 * PARAMS["sigma"] ** 2) * dt
    log_s1 = stepper(np.array([log_s0]), r, q, dt, dw)
    assert log_s1[0] == pytest.approx(expected_log_s1)
