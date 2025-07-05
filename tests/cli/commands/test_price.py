from unittest.mock import MagicMock, patch

import pandas as pd
from typer.testing import CliRunner

from optpricing.cli.main import app

runner = CliRunner()


@patch("optpricing.cli.commands.price.select_fastest_technique")
@patch("optpricing.cli.commands.price.fit_rate_and_dividend", return_value=(0.05, 0.01))
@patch(
    "optpricing.cli.commands.price.get_live_option_chain",
    return_value=pd.DataFrame([{"spot_price": 150.0, "optionType": "call"}]),
)
@patch("optpricing.cli.commands.price.get_live_dividend_yield", return_value=0.01)
def test_price_command(mock_get_div, mock_get_chain, mock_fit_rate, mock_select_tech):
    """
    Tests the 'price' command orchestrates pricing correctly.
    """
    # Mock the technique that gets selected
    mock_technique = MagicMock()
    mock_technique.price.return_value.price = 10.50
    mock_technique.delta.return_value = 0.55
    mock_technique.gamma.return_value = 0.02
    mock_technique.vega.return_value = 0.25
    mock_select_tech.return_value = mock_technique

    result = runner.invoke(
        app,
        [
            "price",
            "--ticker",
            "AAPL",
            "--strike",
            "150",
            "--maturity",
            "2025-12-31",
            "--model",
            "BSM",
            "--param",
            "sigma=0.22",
        ],
    )

    assert result.exit_code == 0
    mock_get_chain.assert_called_once_with("AAPL")
    mock_fit_rate.assert_called_once()
    mock_select_tech.assert_called_once()

    # Check that the final output contains the mocked price
    assert "Price: 10.5000" in result.stdout
    assert "Delta: 0.5500" in result.stdout
    assert "Gamma: 0.0200" in result.stdout
    assert "Vega: 0.2500" in result.stdout
