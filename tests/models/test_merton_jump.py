import numpy as np
import pytest

from optpricing.models import MertonJumpModel

# Common parameters for tests
PARAMS = {
    "sigma": 0.2,
    "lambda": 0.5,
    "mu_j": -0.1,
    "sigma_j": 0.15,
    "max_sum_terms": 100,
}
PRICING_KWARGS = {"spot": 100, "strike": 105, "r": 0.05, "q": 0.01, "t": 1.0}


@pytest.fixture
def model():
    """Provides a MertonJumpModel instance."""
    return MertonJumpModel(params=PARAMS)


def test_parameter_validation():
    """
    Tests that Merton rejects invalid parameters.
    """
    with pytest.raises(ValueError, match="missing required parameters"):
        MertonJumpModel(params={})
    with pytest.raises(ValueError, match="parameters must be positive: lambda"):
        MertonJumpModel(params={**PARAMS, "lambda": -0.1})


def test_closed_form_price(model):
    """
    Tests the closed-form price against a known 'golden' value.
    Value from a trusted online calculator for Merton Jump model.
    """
    call_price = model.price_closed_form(**PRICING_KWARGS, call=True)
    put_price = model.price_closed_form(**PRICING_KWARGS, call=False)
    assert call_price == pytest.approx(8.6710, abs=1e-3)
    assert put_price == pytest.approx(9.54514, abs=1e-3)


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


def test_characteristic_function(model):
    """
    Tests the characteristic function at u=0, where it should be 1.
    """
    cf = model.cf(**PRICING_KWARGS)
    assert cf(0) == pytest.approx(1.0)


def test_sde_stepper(model):
    """
    Tests the SDE stepper for a single step with no randomness and no jumps.
    """
    stepper = model.get_sde_sampler()
    log_s0 = np.log(PRICING_KWARGS["spot"])
    r, q, dt = 0.05, 0.01, 0.01
    dw = np.array([0.0])
    jump_counts = np.array([0])  # No jumps
    rng = np.random.default_rng(0)

    k = np.exp(PARAMS["mu_j"] + 0.5 * PARAMS["sigma_j"] ** 2) - 1
    compensator = PARAMS["lambda"] * k
    expected_drift = (r - q - 0.5 * PARAMS["sigma"] ** 2 - compensator) * dt
    expected_log_s1 = log_s0 + expected_drift

    log_s1 = stepper(np.array([log_s0]), r, q, dt, dw, jump_counts, rng)
    assert log_s1[0] == pytest.approx(expected_log_s1)
