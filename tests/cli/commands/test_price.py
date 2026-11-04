from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
from typer.testing import CliRunner

from optpricing.cli.main import app

runner = CliRunner()

# The command rejects maturities in the past, so the date must be relative to today.
FUTURE_MATURITY = (date.today() + timedelta(days=365)).isoformat()

# A schema-v2 live chain: spot, rate and dividend yield travel with the data.
LIVE_CHAIN = pd.DataFrame(
    [
        {
            "spot_price": 150.0,
            "optionType": "call",
            "strike": 150.0,
            "maturity": 1.0,
            "marketPrice": 10.0,
            "risk_free_rate": 0.05,
            "dividend_yield": 0.01,
            "ticker": "AAPL",
            "snapshot_date": "2025-06-16",
            "as_of": pd.Timestamp("2025-06-16", tz="UTC"),
        }
    ]
)


@patch("optpricing.cli.commands.price.select_fastest_technique")
@patch("optpricing.cli.commands.price.get_live_option_chain", return_value=LIVE_CHAIN)
def test_price_command(mock_get_chain, mock_select_tech):
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
            FUTURE_MATURITY,
            "--model",
            "BSM",
            "--param",
            "sigma=0.22",
        ],
    )

    assert result.exit_code == 0, result.output
    mock_get_chain.assert_called_once_with("AAPL")
    mock_select_tech.assert_called_once()
    assert "Spot 150.00 | Implied dividend 1.0000% | r 5.0000%" in result.stdout

    # Check that the final output contains the mocked price
    assert "Price: 10.5000" in result.stdout
    assert "Delta: 0.5500" in result.stdout
    assert "Gamma: 0.0200" in result.stdout
    assert "Vega: 0.2500" in result.stdout
    # the technique received the chain's market parameters
    _, stock, _, rate = mock_technique.price.call_args.args
    assert stock.spot == 150.0 and stock.dividend == 0.01 and rate.get_rate() == 0.05


@patch("optpricing.cli.commands.price.get_live_option_chain", return_value=LIVE_CHAIN)
def test_price_command_model_alias_and_errors(mock_get_chain):
    ok = runner.invoke(
        app,
        [
            "price",
            "-t",
            "AAPL",
            "-k",
            "150",
            "-T",
            FUTURE_MATURITY,
            "-m",
            "variance gamma",
        ],
    )
    assert ok.exit_code == 0, ok.output
    assert "Price:" in ok.output

    bad_model = runner.invoke(
        app, ["price", "-t", "AAPL", "-k", "150", "-T", FUTURE_MATURITY, "-m", "Nope"]
    )
    assert bad_model.exit_code == 1
    past = runner.invoke(app, ["price", "-t", "AAPL", "-k", "150", "-T", "2020-01-01"])
    assert past.exit_code == 1
