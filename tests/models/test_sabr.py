import math

import numpy as np
import pytest

from optpricing.models import SABRModel
from optpricing.models.sabr import hagan_lognormal_vol

# Common parameters for tests (alpha = initial vol, nu = vol of vol)
PARAMS = {"alpha": 0.5, "beta": 0.8, "rho": -0.6, "nu": 0.5}


@pytest.fixture
def model():
    """Provides a SABRModel instance."""
    return SABRModel(params=PARAMS)


def test_parameter_validation():
    """
    Tests that SABR rejects invalid parameters.
    """
    with pytest.raises(ValueError, match="missing required parameters: beta"):
        SABRModel(params={"alpha": 0.5, "rho": -0.6, "nu": 0.5})
    with pytest.raises(ValueError, match="parameters must be positive: alpha"):
        SABRModel(params={"alpha": -0.5, "beta": 0.8, "rho": -0.6, "nu": 0.5})
    with pytest.raises(ValueError, match="parameters must be positive: nu"):
        SABRModel(params={"alpha": 0.5, "beta": 0.8, "rho": -0.6, "nu": 0.0})
    with pytest.raises(ValueError, match="parameter 'beta' must be in"):
        SABRModel(params={"alpha": 0.5, "beta": 1.1, "rho": -0.6, "nu": 0.5})
    with pytest.raises(ValueError, match="parameter 'rho' must be in"):
        SABRModel(params={"alpha": 0.5, "beta": 0.8, "rho": -1.0, "nu": 0.5})
    with pytest.raises(ValueError, match="'shift' must be non-negative"):
        SABRModel(params={**PARAMS, "shift": -0.01})


def test_legacy_three_parameter_form_is_mapped_to_nu_with_a_warning():
    """Before v2.5 `alpha` was the vol of vol and doubled as the initial vol."""
    with pytest.warns(DeprecationWarning, match="Interpreting nu=alpha"):
        legacy = SABRModel(params={"alpha": 0.5, "beta": 0.8, "rho": -0.6})
    assert legacy.params == PARAMS


def test_not_implemented_methods(model):
    """
    Tests that methods not applicable to SABR raise NotImplementedError.
    """
    with pytest.raises(NotImplementedError):
        model.get_sde_sampler()
    with pytest.raises(NotImplementedError):
        model.cf()
    with pytest.raises(NotImplementedError):
        model.get_pde_coeffs()


# QuantLib 1.44 `sabrVolatility(strike, forward, 1.0, alpha, beta, nu, rho)`,
# forward = 100, alpha = 0.25 * F^(1-beta).
HAGAN_REFERENCES = [
    # beta, rho, nu, alpha, strike, implied vol
    (0.0, -0.6, 0.5, 25.0, 70.0, 0.3599310124),
    (0.0, -0.6, 0.5, 25.0, 100.0, 0.2530468750),
    (0.0, 0.3, 0.8, 25.0, 140.0, 0.2907548146),
    (0.5, -0.6, 0.5, 2.5, 70.0, 0.3314057362),
    (0.5, 0.0, 0.3, 2.5, 100.0, 0.2520377604),
    (0.5, 0.3, 0.8, 2.5, 140.0, 0.3118133651),
    (0.8, -0.6, 0.5, 0.6279716079, 70.0, 0.3157639416),
    (0.8, -0.6, 0.5, 0.6279716079, 90.0, 0.2676699416),
    (0.8, -0.6, 0.5, 0.6279716079, 100.0, 0.2486718750),
    (0.8, -0.6, 0.5, 0.6279716079, 110.0, 0.2329562577),
    (0.8, -0.6, 0.5, 0.6279716079, 140.0, 0.2059394840),
    (1.0, -0.6, 0.5, 0.25, 70.0, 0.3058592956),
    (1.0, 0.0, 0.3, 0.25, 100.0, 0.2518750000),
    (1.0, 0.3, 0.8, 0.25, 140.0, 0.3347360575),
]


@pytest.mark.parametrize("beta, rho, nu, alpha, strike, ref", HAGAN_REFERENCES)
def test_hagan_implied_vol_matches_quantlib(beta, rho, nu, alpha, strike, ref):
    vol = hagan_lognormal_vol(
        100.0, strike, 1.0, alpha=alpha, beta=beta, rho=rho, nu=nu
    )
    assert vol == pytest.approx(ref, abs=1e-7)


def test_hagan_implied_vol_is_continuous_at_the_money():
    """z/x(z) -> 1 as K -> F; the series branch must join the closed form."""
    vols = hagan_lognormal_vol(
        100.0,
        np.array([99.999, 99.99999, 100.0, 100.00001, 100.001]),
        1.0,
        alpha=0.5,
        beta=0.8,
        rho=-0.6,
        nu=0.5,
    )
    assert np.all(np.isfinite(vols))
    assert np.max(np.abs(np.diff(vols))) < 1e-5


def test_shifted_sabr_matches_quantlib():
    """QuantLib `shiftedSabrVolatility(K, F=0.01, T=2, alpha=0.02, beta=0.5,
    nu=0.4, rho=-0.3, shift=0.03)`."""
    model = SABRModel(
        params={"alpha": 0.02, "beta": 0.5, "rho": -0.3, "nu": 0.4, "shift": 0.03}
    )
    for strike, ref in [
        (-0.01, 0.1967849957),
        (0.0, 0.1380090660),
        (0.01, 0.1020275000),
        (0.03, 0.1072296602),
    ]:
        assert model.implied_volatility(0.01, strike, 2.0) == pytest.approx(
            ref, abs=1e-9
        )


def test_closed_form_is_black76_with_hagan_vol(model):
    S, K, r, q, T = 100.0, 105.0, 0.05, 0.01, 1.0
    forward = S * math.exp((r - q) * T)
    sigma = model.implied_volatility(forward, K, T)
    from scipy.stats import norm

    d1 = (math.log(forward / K) + 0.5 * sigma**2 * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    call = math.exp(-r * T) * (forward * norm.cdf(d1) - K * norm.cdf(d2))
    put = math.exp(-r * T) * (K * norm.cdf(-d2) - forward * norm.cdf(-d1))
    assert model.price_closed_form(
        spot=S, strike=K, r=r, q=q, t=T, call=True
    ) == pytest.approx(call, rel=1e-12)
    assert model.price_closed_form(
        spot=S, strike=K, r=r, q=q, t=T, call=False
    ) == pytest.approx(put, rel=1e-12)


def test_implied_vol_rejects_non_positive_shifted_inputs():
    model = SABRModel(params={"alpha": 0.02, "beta": 0.5, "rho": -0.3, "nu": 0.4})
    with pytest.raises(ValueError, match="must be positive"):
        model.implied_volatility(0.01, -0.01, 1.0)
