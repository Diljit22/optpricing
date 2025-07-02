from __future__ import annotations

from typing import Annotated

import pandas as pd
import typer

from optpricing.data import get_live_dividend_yield, get_live_option_chain
from optpricing.parity import ImpliedRateModel

__doc__ = """
CLI commands for miscellaneous financial tools.
"""


def get_implied_rate(
    ticker: Annotated[
        str, typer.Option("--ticker", "-t", help="Stock ticker for the option pair.")
    ],
    strike: Annotated[
        float, typer.Option("--strike", "-k", help="Strike price of the option pair.")
    ],
    maturity: Annotated[
        str,
        typer.Option("--maturity", "-T", help="Maturity date in YYYY-MM-DD format."),
    ],
):
    """Calculates the implied risk-free rate from a live call-put pair."""
    typer.echo(
        f"Fetching live prices for {ticker} {strike} options expiring {maturity}..."
    )

    live_chain = get_live_option_chain(ticker)
    if live_chain is None or live_chain.empty:
        typer.secho(
            f"Error: Could not fetch live option chain for {ticker}.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)

    q = get_live_dividend_yield(ticker)

    maturity_dt = pd.to_datetime(maturity).date()
    chain_for_expiry = live_chain[live_chain["expiry"].dt.date == maturity_dt]

    call_option = chain_for_expiry[
        (chain_for_expiry["strike"] == strike)
        & (chain_for_expiry["optionType"] == "call")
    ]
    put_option = chain_for_expiry[
        (chain_for_expiry["strike"] == strike)
        & (chain_for_expiry["optionType"] == "put")
    ]

    if call_option.empty or put_option.empty:
        typer.secho(
            f"Error: Did not find both: call & put for strike {strike} on {maturity}.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)

    call_price = call_option["marketPrice"].iloc[0]
    put_price = put_option["marketPrice"].iloc[0]

    spot_price = call_option["spot_price"].iloc[0]
    maturity_years = call_option["maturity"].iloc[0]

    pair_msg = (
        f"Found Pair -> Call Price: {call_price:.2f}, Put Price: {put_price:.2f}, "
        f"Spot: {spot_price:.2f}"
    )
    typer.echo(pair_msg)

    implied_rate_model = ImpliedRateModel(params={})
    try:
        implied_r = implied_rate_model.price_closed_form(
            call_price=call_price,
            put_price=put_price,
            spot=spot_price,
            strike=strike,
            t=maturity_years,
            q=q,
        )
        typer.secho(
            f"\nImplied Risk-Free Rate (r): {implied_r:.4%}", fg=typer.colors.GREEN
        )
    except Exception as e:
        typer.secho(f"\nError calculating implied rate: {e}", fg=typer.colors.RED)
