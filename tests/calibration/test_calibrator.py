from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from optpricing.atoms import Rate, Stock
from optpricing.calibration import Calibrator
from optpricing.models import BSMModel


@pytest.fixture
def setup():
    """Provides a standard setup for calibrator tests."""
    model = BSMModel()
    technique = MagicMock()
    technique.price.return_value = MagicMock(price=10.5)
    market_data = pd.DataFrame(
        {
            "strike": [100],
            "maturity": [1.0],
            "optionType": ["call"],
            "marketPrice": [10.0],
        }
    )
    stock = Stock(spot=100)
    rate = Rate(rate=0.05)

    calibrator = Calibrator(model, market_data, stock, rate)
    calibrator.technique = technique
    return calibrator


def test_objective_function(setup):
    """
    Tests that the objective function correctly calculates the squared error.
    """
    calibrator = setup

    # Patch print within the objective function to avoid formatting MagicMocks
    with patch("optpricing.calibration.calibrator.print"):
        error = calibrator._objective_function(
            params_to_fit_values=[0.2], params_to_fit_names=["sigma"], frozen_params={}
        )

    # model.price returns 10.5, marketPrice is 10.0
    assert error == pytest.approx(0.2030255)


@patch("optpricing.calibration.calibrator.print")
@patch("optpricing.calibration.calibrator.minimize")
def test_fit_multivariate(mock_minimize, mock_print, setup):
    """
    Tests that fit correctly calls the multivariate optimizer.
    """
    calibrator = setup

    fake_result = MagicMock()
    fake_result.x = [0.2, 0.1]
    fake_result.fun = 0.1234
    mock_minimize.return_value = fake_result

    initial_guess = {"sigma": 0.2, "other": 0.1}
    bounds = {"sigma": (0.01, 1), "other": (0, 1)}

    result = calibrator.fit(initial_guess, bounds)

    mock_minimize.assert_called_once()
    _, kwargs = mock_minimize.call_args

    assert kwargs["x0"] == [0.2, 0.1]
    assert kwargs["bounds"] == [(0.01, 1), (0, 1)]

    assert result["sigma"] == pytest.approx(0.2)
    assert result["other"] == pytest.approx(0.1)


@patch("optpricing.calibration.calibrator.print")
@patch("optpricing.calibration.calibrator.minimize_scalar")
def test_fit_scalar(mock_minimize_scalar, mock_print, setup):
    """
    Tests that fit correctly calls the scalar optimizer for a single parameter.
    """
    calibrator = setup

    fake_result = MagicMock()
    fake_result.x = 0.42
    fake_result.fun = 0.5678
    mock_minimize_scalar.return_value = fake_result

    initial_guess = {"sigma": 0.2}
    bounds = {"sigma": (0.01, 1)}

    result = calibrator.fit(initial_guess, bounds)

    mock_minimize_scalar.assert_called_once()
    _, kwargs = mock_minimize_scalar.call_args
    assert kwargs["bounds"] == (0.01, 1)

    assert result["sigma"] == pytest.approx(0.42)
