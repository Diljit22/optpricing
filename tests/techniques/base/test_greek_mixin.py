from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from optpricing.atoms import Option, OptionType, Rate, Stock
from optpricing.models import BaseModel
from optpricing.techniques.base import BaseTechnique, GreekMixin, PricingResult


class AnalyticTestModel(BaseModel):
    """
    Minimal, concrete implementation of BaseModel for testing. Provides dummy
    implementations for all abstract methods to allow instantiation.
    """

    def __init__(self, params: dict[str, Any]):
        self._params = params

    @property
    def params(self) -> dict[str, Any]:
        return self._params

    def with_params(self, **kwargs: Any) -> AnalyticTestModel:
        new_params = self.params.copy()
        new_params.update(kwargs)
        return AnalyticTestModel(params=new_params)

    def _validate_params(self) -> None:
        pass

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__) and self.params == other.params

    def __hash__(self) -> int:
        return hash(tuple(sorted(self.params.items())))

    def _closed_form_impl(self, **kwargs: Any) -> Any:
        raise NotImplementedError

    def _cf_impl(self, **kwargs: Any) -> Any:
        raise NotImplementedError

    def _sde_impl(self, **kwargs: Any) -> Any:
        raise NotImplementedError

    def _pde_impl(self, **kwargs: Any) -> Any:
        raise NotImplementedError


class AnalyticTestPricer(BaseTechnique, GreekMixin):
    """
    Test pricer using a simple formula with known analytical derivatives
    to verify the correctness of the numerical methods in GreekMixin.
    Price(S, K, T, r, sigma) = S * sigma - K * exp(-r * T)
    """

    def __init__(self, use_crn: bool = False):
        if use_crn:
            self.rng = np.random.default_rng(0)

    def price(
        self,
        option: Option,
        stock: Stock,
        model: AnalyticTestModel,
        rate: Rate,
        **kwargs: Any,
    ) -> PricingResult:
        S = stock.spot
        K = option.strike
        T = option.maturity
        r = rate.get_rate(T)
        sigma = model.params["sigma"]
        price_val = S * sigma - K * np.exp(-r * T)
        return PricingResult(price=price_val)

    # Analytical Greeks for the formula above
    def analytical_delta(self, model: AnalyticTestModel) -> float:
        return model.params["sigma"]

    def analytical_gamma(self) -> float:
        return 0.0

    def analytical_vega(self, stock: Stock) -> float:
        return stock.spot

    def analytical_theta(self, option: Option, rate: Rate) -> float:
        r = rate.get_rate(option.maturity)
        return -option.strike * r * np.exp(-r * option.maturity)

    def analytical_rho(self, option: Option, rate: Rate) -> float:
        r = rate.get_rate(option.maturity)
        return option.strike * option.maturity * np.exp(-r * option.maturity)


@pytest.fixture(params=[True, False], ids=["WithCRN", "WithoutCRN"])
def setup(request):
    """Provides a standard setup for Greek tests."""
    use_crn = request.param
    technique = AnalyticTestPricer(use_crn=use_crn)
    option = Option(strike=100, maturity=1.0, option_type=OptionType.CALL)
    stock = Stock(spot=105)
    model = AnalyticTestModel(params={"sigma": 0.2})
    rate = Rate(rate=0.05)
    return technique, option, stock, model, rate


def test_delta_correctness(setup):
    """
    Tests that the numerical delta matches the analytical delta.
    """
    technique, option, stock, model, rate = setup
    numerical_delta = technique.delta(option, stock, model, rate)
    analytical_delta = technique.analytical_delta(model)
    assert np.isclose(numerical_delta, analytical_delta)


def test_gamma_correctness(setup):
    """
    Tests that the numerical gamma matches the analytical gamma.
    """
    technique, option, stock, model, rate = setup
    numerical_gamma = technique.gamma(option, stock, model, rate)
    analytical_gamma = technique.analytical_gamma()
    assert np.isclose(numerical_gamma, analytical_gamma, atol=1e-7)


def test_vega_correctness(setup):
    """
    Tests that the numerical vega matches the analytical vega.
    """
    technique, option, stock, model, rate = setup
    numerical_vega = technique.vega(option, stock, model, rate)
    analytical_vega = technique.analytical_vega(stock)
    assert np.isclose(numerical_vega, analytical_vega)


def test_rho_correctness(setup):
    """
    Tests that the numerical rho matches the analytical rho.
    """
    technique, option, stock, model, rate = setup
    numerical_rho = technique.rho(option, stock, model, rate)
    analytical_rho = technique.analytical_rho(option, rate)
    assert np.isclose(numerical_rho, analytical_rho)


def test_vega_no_sigma_returns_nan(setup):
    """
    Tests that vega returns nan if the model has no 'sigma' parameter.
    """
    technique, option, stock, _, rate = setup
    no_sigma_model = AnalyticTestModel(params={"other_param": 0.1})
    vega = technique.vega(option, stock, no_sigma_model, rate)
    assert np.isnan(vega)
