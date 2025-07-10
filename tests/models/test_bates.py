import numpy as np
import pytest

from optpricing.models import BatesModel

# Common parameters for tests
PARAMS = {
    "kappa": 2.0,
    "theta": 0.04,
    "rho": -0.7,
    "vol_of_vol": 0.5,
    "lambda": 0.5,
    "mu_j": -0.1,
    "sigma_j": 0.15,
}
PRICING_KWARGS = {"spot": 100, "r": 0.05, "q": 0.01, "t": 1.0, "v0": 0.04}


@pytest.fixture
def model():
    """Provides a BatesModel instance."""
    return BatesModel(params=PARAMS)


def test_parameter_validation():
    """
    Tests that Bates rejects invalid parameters.
    """
    with pytest.raises(ValueError, match="parameters must be positive: vol_of_vol"):
        BatesModel(params={**PARAMS, "vol_of_vol": -0.1})
    with pytest.raises(ValueError, match="parameter 'rho' must be in"):
        BatesModel(params={**PARAMS, "rho": -1.5})


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
    v0 = PRICING_KWARGS["v0"]
    r, q, dt = 0.05, 0.01, 0.01
    dw_s, dw_v = np.array([0.0]), np.array([0.0])
    jump_counts = np.array([0])
    rng = np.random.default_rng(0)

    k = np.exp(PARAMS["mu_j"] + 0.5 * PARAMS["sigma_j"] ** 2) - 1
    compensator = PARAMS["lambda"] * k
    expected_s_drift = (r - q - 0.5 * v0 - compensator) * dt
    expected_log_s1 = log_s0 + expected_s_drift
    expected_v1 = v0 + PARAMS["kappa"] * (PARAMS["theta"] - v0) * dt

    log_s1, v1 = stepper(
        np.array([log_s0]), np.array([v0]), r, q, dt, dw_s, dw_v, jump_counts, rng
    )

    assert log_s1[0] == pytest.approx(expected_log_s1)
    assert v1[0] == pytest.approx(expected_v1)
