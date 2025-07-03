from __future__ import annotations

from typing import Annotated

import pandas as pd
import typer

from optpricing.atoms import Option, OptionType, Rate, Stock
from optpricing.calibration import fit_rate_and_dividend
from optpricing.calibration.technique_selector import select_fastest_technique
from optpricing.data import get_live_dividend_yield, get_live_option_chain
from optpricing.models import BSMModel
from optpricing.techniques import LeisenReimerTechnique
from optpricing.workflows.configs import ALL_MODEL_CONFIGS

__doc__ = """
CLI command for on-demand option pricing.
"""


def price(
    ticker: Annotated[
        str, typer.Option("--ticker", "-t", help="Stock ticker for the option.")
    ],
    strike: Annotated[
        float, typer.Option("--strike", "-k", help="Strike price of the option.")
    ],
    maturity: Annotated[
        str,
        typer.Option("--maturity", "-T", help="Maturity date in YYYY-MM-DD format."),
    ],
    option_type: Annotated[
        str, typer.Option("--type", help="Option type: 'call' or 'put'.")
    ] = "call",
    style: Annotated[
        str,
        typer.Option(
            "--style",
            help="Option exercise style: 'american' or 'european'.",
            case_sensitive=False,
        ),
    ] = "european",
    model: Annotated[
        str, typer.Option("--model", "-m", help="The model to use for pricing.")
    ] = "BSM",
    param: Annotated[
        list[str] | None,
        typer.Option(
            "--param",
            help="Set a model parameter (e.g., 'sigma=0.2'). Can use multiple times.",
        ),
    ] = None,
):
    """Prices a single option using live market data and user model parameters."""
    msg = (
        f"Pricing {ticker} {strike} {option_type.upper()} expiring {maturity} "
        f"using {model} model..."
    )
    typer.echo(msg)

    model_params = {}
    if param:
        for p in param:
            try:
                key, value = p.split("=")
                model_params[key.strip()] = float(value)
            except ValueError:
                typer.secho(
                    f"Invalid format for parameter: '{p}'. Use 'key=value'.",
                    fg=typer.colors.RED,
                )
                raise typer.Exit(code=1)

    typer.echo("Fetching live market data...")
    live_chain = get_live_option_chain(ticker)
    if live_chain is None or live_chain.empty:
        typer.secho(
            f"Error: Could not fetch live option chain for {ticker}.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)

    q = get_live_dividend_yield(ticker)
    spot = live_chain["spot_price"].iloc[0]
    calls = live_chain[live_chain["optionType"] == "call"]
    puts = live_chain[live_chain["optionType"] == "put"]

    r, _ = fit_rate_and_dividend(calls, puts, spot, q_fixed=q)
    typer.echo(
        f"Live Data -> Spot: {spot:.2f}, Known Dividend: {q:.4%}, Implied Rate: {r:.4%}"
    )

    stock = Stock(spot=spot, dividend=q)
    rate = Rate(rate=r)
    maturity_years = (pd.to_datetime(maturity) - pd.Timestamp.now()).days / 365.25
    option = Option(
        strike=strike,
        maturity=maturity_years,
        option_type=OptionType[option_type.upper()],
    )

    model_class = ALL_MODEL_CONFIGS[model]["model_class"]
    model_instance = model_class(params=model_params)

    is_american_flag = style.lower() == "american"

    if is_american_flag:
        # Check if the selected model is BSM, which is the only one
        # that currently supports lattice methods.
        if isinstance(model_instance, BSMModel):
            # Force the use of a lattice technique for American pricing
            technique = LeisenReimerTechnique(is_american=True)
            typer.echo("American style requested. Using Leisen-Reimer lattice.")
        else:
            # User wants American, but the model doesn't support it.
            # Warn and fall back to the fastest European method.
            technique = select_fastest_technique(model_instance)
            typer.secho(
                f"Warning: {model_instance.name} does not support American pricing.",
                fg=typer.colors.YELLOW,
            )
            typer.secho(
                f"         Pricing as European using {technique.__class__.__name__}.",
                fg=typer.colors.YELLOW,
            )
    else:
        # Default European case
        technique = select_fastest_technique(model_instance)

    pricing_kwargs = model_params.copy()

    price_result = technique.price(
        option,
        stock,
        model_instance,
        rate,
        **pricing_kwargs,
    )
    delta = technique.delta(
        option,
        stock,
        model_instance,
        rate,
        **pricing_kwargs,
    )
    gamma = technique.gamma(
        option,
        stock,
        model_instance,
        rate,
        **pricing_kwargs,
    )
    vega = technique.vega(
        option,
        stock,
        model_instance,
        rate,
        **pricing_kwargs,
    )

    typer.secho("\n--- Pricing Results ---", fg=typer.colors.CYAN)
    typer.echo(f"Price: {price_result.price:.4f}")
    typer.echo(f"Delta: {delta:.4f}")
    typer.echo(f"Gamma: {gamma:.4f}")
    typer.echo(f"Vega:  {vega:.4f}")
