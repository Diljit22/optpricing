from __future__ import annotations

from typing import Annotated, Any

import pandas as pd
import typer

from optpricing.atoms import Option, OptionType, Rate, Stock
from optpricing.calibration import fit_rate_and_dividend
from optpricing.calibration.technique_selector import select_fastest_technique
from optpricing.data import get_live_dividend_yield, get_live_option_chain
from optpricing.models import BSMModel
from optpricing.techniques import (
    AmericanMonteCarloTechnique,
    CRRTechnique,
    LeisenReimerTechnique,
    MonteCarloTechnique,
)
from optpricing.workflows.configs import ALL_MODEL_CONFIGS

__doc__ = """
CLI command for on-demand option pricing.
"""


def _err(msg: str) -> None:
    """Helper to print a formatted error message and exit."""
    typer.secho(msg, fg=typer.colors.RED, err=True)
    raise typer.Exit(code=1)


def price(
    ticker: Annotated[
        str,
        typer.Option(
            "--ticker",
            "-t",
            help="Stock ticker.",
        ),
    ],
    strike: Annotated[
        float,
        typer.Option(
            "--strike",
            "-k",
            help="Strike price.",
        ),
    ],
    maturity: Annotated[
        str,
        typer.Option(
            "--maturity",
            "-T",
            help="Maturity YYYY-MM-DD.",
        ),
    ],
    option_type: Annotated[
        str,
        typer.Option(
            "--type",
            help="call|put",
            case_sensitive=False,
        ),
    ] = "call",
    style: Annotated[
        str,
        typer.Option(
            "--style",
            help="european|american",
            case_sensitive=False,
        ),
    ] = "european",
    model: Annotated[
        str,
        typer.Option(
            "--model",
            "-m",
            help="Model key.",
        ),
    ] = "BSM",
    technique: Annotated[
        str | None,
        typer.Option(
            "--technique",
            "-x",
            help="Force technique key (e.g., mc, crr).",
            case_sensitive=False,
        ),
    ] = None,
    param: Annotated[
        list[str] | None,
        typer.Option(
            "--param",
            help="Repeat: key=value (e.g. sigma=0.2)",
            show_default=False,
        ),
    ] = None,
):
    """
    Prices a single option using live market data and specified parameters.
    """

    def _parse_params(param_list: list[str] | None) -> dict[str, float]:
        params: dict[str, float] = {}
        if not param_list:
            return params
        for raw in param_list:
            if "=" not in raw:
                _err(f"Invalid --param '{raw}'. Expected format key=value.")
            key, value = (tok.strip() for tok in raw.split("=", 1))
            try:
                params[key] = float(value)
            except ValueError:
                _err(f"Could not parse float from '{value}' (param '{key}').")
        return params

    def _select_technique() -> Any:
        technique_map = {
            "MC": MonteCarloTechnique,
            "AMERICAN_MC": AmericanMonteCarloTechnique,
            "CRR": CRRTechnique,
            "LR": LeisenReimerTechnique,
        }
        if technique:
            key = technique.upper()
            if key not in technique_map:
                _err(
                    f"Technique '{technique}' not recognised. "
                    f"Choices: {', '.join(technique_map)}"
                )
            try:
                return technique_map[key](is_american=is_american)
            except TypeError:
                return technique_map[key]()

        if is_american:
            if isinstance(model_instance, BSMModel):
                typer.echo("American style detected. Using Leisen-Reimer lattice.")
                return LeisenReimerTechnique(is_american=True)
            typer.echo("American style detected. Falling back to LSMC (American MC).")
            return AmericanMonteCarloTechnique()

        fastest = select_fastest_technique(model_instance)
        _tech = fastest.__class__.__name__
        typer.echo(f"European style. Auto-selected fastest technique: {_tech}.")
        return fastest

    # Parse & Validate Inputs
    typer.echo(
        f"Pricing a {style} {ticker} {option_type.upper()} "
        f"(K={strike}) exp {maturity} using {model}..."
    )
    model_params = _parse_params(param)
    is_american = style.lower() == "american"

    # Live-Data Fetch
    typer.echo("Fetching live option chain...")
    live_chain = get_live_option_chain(ticker)
    if live_chain is None or live_chain.empty:
        _err(f"No live option chain found for {ticker}.")

    q_div = get_live_dividend_yield(ticker)
    spot = live_chain["spot_price"].iloc[0]
    calls = live_chain[live_chain["optionType"] == "call"]
    puts = live_chain[live_chain["optionType"] == "put"]
    r_rate, _ = fit_rate_and_dividend(calls, puts, spot, q_fixed=q_div)
    typer.echo(
        f"Live Data: Spot {spot:.2f} | Dividend {q_div:.4%} | Implied r {r_rate:.4%}"
    )

    # Build Atoms & Model
    stock = Stock(spot=spot, dividend=q_div)
    rate = Rate(rate=r_rate)
    maturity_years = (
        pd.to_datetime(maturity) - pd.Timestamp.utcnow().tz_localize(None)
    ).days / 365.25
    if maturity_years <= 0:
        _err("Maturity date must be in the future.")

    option = Option(
        strike=strike,
        maturity=maturity_years,
        option_type=OptionType[option_type.upper()],
    )

    try:
        model_cls = ALL_MODEL_CONFIGS[model]["model_class"]
    except KeyError:
        _err(f"Model '{model}' not recognised in ALL_MODEL_CONFIGS.")

    # Merge default params with user-provided params
    full_params = (
        model_cls.default_params.copy() if hasattr(model_cls, "default_params") else {}
    )
    full_params.update(model_params)
    model_instance = model_cls(params=full_params)

    # Technique Selection & Pricing
    technique_instance = _select_technique()

    price_result = technique_instance.price(
        option,
        stock,
        model_instance,
        rate,
        **full_params,
    )

    typer.secho("\n── Pricing Results " + "─" * 38, fg=typer.colors.CYAN)
    typer.echo(f"Price: {price_result.price:.4f}")

    # Greeks
    for greek_name in [
        "Delta",
        "Gamma",
        "Vega",
        "Theta",
        "Rho",
    ]:
        greek_func = getattr(
            technique_instance,
            greek_name.lower(),
            None,
        )
        if callable(greek_func):
            try:
                value = greek_func(
                    option,
                    stock,
                    model_instance,
                    rate,
                    **full_params,
                )
                if isinstance(value, int | float):
                    typer.echo(f"{greek_name}: {value:.4f}")
            except NotImplementedError:
                continue
