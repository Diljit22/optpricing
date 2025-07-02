from unittest.mock import patch

import pandas as pd
from typer.testing import CliRunner

from optpricing.cli.main import app

runner = CliRunner()


@patch("optpricing.cli.commands.tools.ImpliedRateModel")
@patch(
    "optpricing.cli.commands.tools.get_live_option_chain",
    return_value=pd.DataFrame(
        [
            {
                "spot_price": 200.0,
                "strike": 200.0,
                "optionType": "call",
                "marketPrice": 15.0,
                "maturity": 1.0,
                "expiry": pd.to_datetime("2025-06-20"),
            },
            {
                "spot_price": 200.0,
                "strike": 200.0,
                "optionType": "put",
                "marketPrice": 12.0,
                "maturity": 1.0,
                "expiry": pd.to_datetime("2025-06-20"),
            },
        ]
    ),
)
@patch("optpricing.cli.commands.tools.get_live_dividend_yield", return_value=0.01)
def test_tools_implied_rate_command(
    mock_get_div, mock_get_chain, mock_implied_rate_model
):
    """
    Tests the 'tools implied-rate' subcommand.
    """
    mock_model_instance = mock_implied_rate_model.return_value
    mock_model_instance.price_closed_form.return_value = 0.0525  # Mocked implied rate

    result = runner.invoke(
        app,
        [
            "tools",
            "implied-rate",
            "--ticker",
            "MSFT",
            "--strike",
            "200",
            "--maturity",
            "2025-06-20",
        ],
    )

    assert result.exit_code == 0
    mock_get_chain.assert_called_once_with("MSFT")
    mock_implied_rate_model.assert_called_once()
    mock_model_instance.price_closed_form.assert_called_once()

    # Check that the final output contains the mocked rate
    assert "Implied Risk-Free Rate (r): 5.2500%" in result.stdout
