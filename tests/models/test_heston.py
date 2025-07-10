import numpy as np
import pytest

from optpricing.models import HestonModel

# Common parameters for tests
PARAMS = {"kappa": 2.0, "theta": 0.04, "rho": -0.7, "vol_of_vol": 0.5}
PRICING_KWARGS = {"spot": 100, "r": 0.05, "q": 0.01, "t": 1.0, "v0": 0.04}


@pytest.fixture
def model():
    """Provides a HestonModel instance."""
    return HestonModel(params=PARAMS)


def test_parameter_validation():
    """
    Tests that Heston rejects invalid parameters.
    """
    with pytest.raises(ValueError, match="parameters must be positive: kappa"):
        HestonModel(params={**PARAMS, "kappa": -1})
    with pytest.raises(ValueError, match="parameter 'rho' must be in"):
        HestonModel(params={**PARAMS, "rho": 1.5})


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
    v0 = PRICING_KWARGS["v0"]
    r, q, dt = 0.05, 0.01, 0.01
    dw_s = np.array([0.0])  # No random shock for spot
    dw_v = np.array([0.0])  # No random shock for variance

    expected_log_s1 = log_s0 + (r - q - 0.5 * v0) * dt
    expected_v1 = v0 + PARAMS["kappa"] * (PARAMS["theta"] - v0) * dt

    log_s1, v1 = stepper(np.array([log_s0]), np.array([v0]), r, q, dt, dw_s, dw_v)

    assert log_s1[0] == pytest.approx(expected_log_s1)
    assert v1[0] == pytest.approx(expected_v1)
