from __future__ import annotations

import time
from typing import Any

import numpy as np
import typer
from rich.console import Console
from rich.table import Table

from optpricing.atoms import Option, OptionType, Rate, Stock
from optpricing.models import (
    BatesModel,
    BSMModel,
    CEVModel,
    CGMYModel,
    HestonModel,
    HyperbolicModel,
    KouModel,
    MertonJumpModel,
    NIGModel,
    SABRJumpModel,
    SABRModel,
    VarianceGammaModel,
)
from optpricing.models.base.base_model import BaseModel
from optpricing.parity.parity_model import ParityModel
from optpricing.techniques import (
    ClosedFormTechnique,
    CRRTechnique,
    FFTTechnique,
    IntegrationTechnique,
    LeisenReimerTechnique,
    MonteCarloTechnique,
    PDETechnique,
    TOPMTechnique,
)

__doc__ = """
A comprehensive benchmark for pricing European options across various models
and numerical techniques. This script compares prices, Greeks, and performance.
"""

app = typer.Typer(
    name="european-benchmark",
    help="Runs benchmark demos for European options.",
    add_completion=False,
    no_args_is_help=True,
)
console = Console()
parity_model = ParityModel(params={})


def profile_all_metrics(
    technique: Any,
    option: Option,
    stock: Stock,
    model: BaseModel,
    rate: Rate,
    **kwargs,
) -> dict[str, tuple[float, float]]:
    """Helper to run and profile all calculations for a given technique."""
    results = {}
    price_val = np.nan

    start = time.perf_counter()
    try:
        price_val = technique.price(option, stock, model, rate, **kwargs).price
    except (NotImplementedError, TypeError, ValueError):
        price_val = np.nan
    end = time.perf_counter()
    results["Price"] = (price_val, end - start)

    greeks_to_calc = {
        "Delta": technique.delta,
        "Gamma": technique.gamma,
        "Vega": technique.vega,
        "Theta": technique.theta,
        "Rho": technique.rho,
    }
    skip_mc_greeks = model.has_jumps and isinstance(technique, MonteCarloTechnique)
    if skip_mc_greeks and isinstance(technique, MonteCarloTechnique):
        console.print(
            "[yellow]  (Skipping unstable MC greeks for jump model)[/yellow]",
            highlight=False,
        )

    for name, func in greeks_to_calc.items():
        start = time.perf_counter()
        greek_val = np.nan
        if not skip_mc_greeks:
            try:
                greek_val = func(option, stock, model, rate, **kwargs)
            except (NotImplementedError, TypeError, ValueError):
                greek_val = np.nan
        end = time.perf_counter()
        results[name] = (greek_val, end - start)

    start = time.perf_counter()
    iv_val = np.nan
    if np.isfinite(price_val):
        try:
            iv_val = technique.implied_volatility(
                option, stock, model, rate, target_price=price_val, **kwargs
            )
        except (NotImplementedError, TypeError, ValueError, RuntimeError):
            iv_val = np.nan
    end = time.perf_counter()
    results["ImpliedVol"] = (iv_val, end - start)
    return results


def run_benchmark(config: dict[str, Any]):
    """Runs and prints a formatted benchmark for a given model configuration."""
    model, stock, rate, techniques, kwargs = (
        config["model_instance"],
        config["stock"],
        config["rate"],
        config["techniques"],
        config.get("kwargs", {}),
    )
    console.rule(f"[bold cyan]{config['model_name']}[/bold cyan]", style="cyan")
    console.print(f"[bold]Model:[/] {model}")
    console.print(f"[bold]Stock:[/] {stock}")
    console.print(f"[bold]Rate:[/]  {rate}")

    all_results: dict[str, dict[str, Any]] = {}
    for option_type in [OptionType.CALL, OptionType.PUT]:
        option = Option(strike=100.0, maturity=1.0, option_type=option_type)
        header = (
            f"OPTION: {option.option_type.value.upper()} | "
            f"K={option.strike} | T={option.maturity}"
        )

        console.rule(header, style="white")

        tech_results = {
            name: profile_all_metrics(tech, option, stock, model, rate, **kwargs)
            for name, tech in techniques.items()
        }
        all_results[option_type.value] = tech_results

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="dim", width=12)
        for tech_name in techniques:
            table.add_column(tech_name, justify="center")

        def _fmt_cell(val_time: tuple[float, float]) -> str:
            price, elapsed = val_time
            return f"{price:.4f}\n[dim]({elapsed:.4f} s)[/dim]"

        metrics = [
            "Price",
            "Delta",
            "Gamma",
            "Vega",
            "Theta",
            "Rho",
            "ImpliedVol",
        ]
        for metric in metrics:
            row_data = [_fmt_cell(tech_results[tech][metric]) for tech in techniques]
            table.add_row(metric, *row_data)
        console.print(table)

    console.print("[bold]Put-Call Parity Errors:[/bold]")
    parity_errors = {}
    for tech_name in techniques:
        call_price = all_results["Call"][tech_name]["Price"][0]
        put_price = all_results["Put"][tech_name]["Price"][0]
        if np.isfinite(call_price) and np.isfinite(put_price):
            expected_put = parity_model._closed_form_impl(
                spot=stock.spot,
                strike=option.strike,
                r=rate.get_rate(option.maturity),
                t=option.maturity,
                call=True,
                option_price=call_price,
                q=stock.dividend,
            )
            parity_errors[tech_name] = put_price - expected_put
        else:
            parity_errors[tech_name] = np.nan

    parity_table = Table(
        show_header=True, header_style="bold magenta", show_edge=False, box=None
    )
    for tech_name in parity_errors:
        parity_table.add_column(tech_name, justify="center")
    parity_table.add_row(*[f"{err:.4e}" for err in parity_errors.values()])
    console.print(parity_table)


def get_benchmark_configs() -> list[dict[str, Any]]:
    """Returns a list of all benchmark configurations."""
    stock = Stock(spot=100.0, dividend=0.01)
    rate = Rate(rate=0.03)
    mc_technique = MonteCarloTechnique(n_paths=50000, n_steps=100, seed=42)
    mc_exact_technique = MonteCarloTechnique(n_paths=50000, seed=42)

    return [
        {
            "model_name": "BLACK-SCHOLES-MERTON",
            "model_instance": BSMModel(params={"sigma": 0.2}),
            "stock": stock,
            "rate": rate,
            "techniques": {
                "Analytic": ClosedFormTechnique(),
                "Integration": IntegrationTechnique(),
                "FFT": FFTTechnique(),
                "PDE": PDETechnique(M=500, N=500),
                "CRR": CRRTechnique(steps=501),
                "LR": LeisenReimerTechnique(steps=501),
                "TOPM": TOPMTechnique(steps=501),
                "MC": mc_technique,
            },
        },
        {
            "model_name": "MERTON JUMP-DIFFUSION",
            "model_instance": MertonJumpModel(
                params={
                    "sigma": 0.2,
                    "lambda": 0.5,
                    "mu_j": -0.1,
                    "sigma_j": 0.15,
                    "max_sum_terms": 100,
                }
            ),
            "stock": stock,
            "rate": rate,
            "techniques": {
                "Closed-Form": ClosedFormTechnique(),
                "Integration": IntegrationTechnique(),
                "FFT": FFTTechnique(),
                "MC": mc_technique,
            },
        },
        {
            "model_name": "HESTON",
            "model_instance": HestonModel(
                params={
                    "v0": 0.04,
                    "kappa": 2.0,
                    "theta": 0.04,
                    "rho": -0.7,
                    "vol_of_vol": 0.5,
                }
            ),
            "stock": stock,
            "rate": rate,
            "kwargs": {"v0": 0.04},
            "techniques": {
                "Integration": IntegrationTechnique(),
                "FFT": FFTTechnique(),
                "MC": mc_technique,
            },
        },
        {
            "model_name": "BATES",
            "model_instance": BatesModel(
                params={
                    "v0": 0.04,
                    "kappa": 2.0,
                    "theta": 0.04,
                    "rho": -0.7,
                    "vol_of_vol": 0.5,
                    "lambda": 0.5,
                    "mu_j": -0.1,
                    "sigma_j": 0.15,
                }
            ),
            "stock": stock,
            "rate": rate,
            "kwargs": {"v0": 0.04},
            "techniques": {
                "Integration": IntegrationTechnique(),
                "FFT": FFTTechnique(),
                "MC": mc_technique,
            },
        },
        {
            "model_name": "KOU",
            "model_instance": KouModel(
                params={
                    "sigma": 0.15,
                    "lambda": 1.0,
                    "p_up": 0.6,
                    "eta1": 10.0,
                    "eta2": 5.0,
                }
            ),
            "stock": stock,
            "rate": rate,
            "techniques": {
                "Integration": IntegrationTechnique(),
                "FFT": FFTTechnique(),
                "MC": mc_technique,
            },
        },
        {
            "model_name": "VARIANCE GAMMA (VG)",
            "model_instance": VarianceGammaModel(
                params={"sigma": 0.2, "nu": 0.1, "theta": -0.14}
            ),
            "stock": stock,
            "rate": rate,
            "techniques": {
                "Integration": IntegrationTechnique(),
                "FFT": FFTTechnique(),
                "MC": mc_exact_technique,
            },
        },
        {
            "model_name": "CGMY",
            "model_instance": CGMYModel(
                params={"C": 0.02, "G": 5.0, "M": 5.0, "Y": 1.2}
            ),
            "stock": stock,
            "rate": rate,
            "techniques": {
                "Integration": IntegrationTechnique(),
                "FFT": FFTTechnique(),
            },
        },
        {
            "model_name": "NORMAL INVERSE GAUSSIAN (NIG)",
            "model_instance": NIGModel(
                params={"alpha": 15.0, "beta": -5.0, "delta": 0.5}
            ),
            "stock": stock,
            "rate": rate,
            "techniques": {
                "Integration": IntegrationTechnique(),
                "FFT": FFTTechnique(),
                "MC": mc_exact_technique,
            },
        },
        {
            "model_name": "HYPERBOLIC",
            "model_instance": HyperbolicModel(
                params={
                    "alpha": 15.0,
                    "beta": -5.0,
                    "delta": 0.5,
                    "mu": 0.0,
                }
            ),
            "stock": stock,
            "rate": rate,
            "techniques": {
                "Integration": IntegrationTechnique(),
                "FFT": FFTTechnique(),
                "MC": mc_exact_technique,
            },
        },
        {
            "model_name": "CONSTANT ELASTICITY OF VARIANCE (CEV)",
            "model_instance": CEVModel(
                params={
                    "sigma": 0.8,
                    "gamma": 0.7,
                }
            ),
            "stock": stock,
            "rate": rate,
            "techniques": {"MC": mc_exact_technique},
        },
        {
            "model_name": "SABR",
            "model_instance": SABRModel(
                params={"alpha": 0.5, "beta": 0.8, "rho": -0.6}
            ),
            "stock": stock,
            "rate": rate,
            "kwargs": {"v0": 0.5},
            "techniques": {"MC": mc_technique},
        },
        {
            "model_name": "SABR JUMP",
            "model_instance": SABRJumpModel(
                params={
                    "alpha": 0.5,
                    "beta": 0.8,
                    "rho": -0.6,
                    "lambda": 0.4,
                    "mu_j": -0.1,
                    "sigma_j": 0.15,
                }
            ),
            "stock": stock,
            "rate": rate,
            "kwargs": {"v0": 0.5},
            "techniques": {"MC": mc_technique},
        },
    ]


@app.command()
def main(
    model: str = typer.Option(
        None,
        "--model",
        "-m",
        help="Run for a specific model (e.g., 'BSM', 'Heston').",
        case_sensitive=False,
    ),
    technique: str = typer.Option(
        None,
        "--technique",
        "-t",
        help="Run for a specific technique (e.g., 'MC', 'FFT').",
        case_sensitive=False,
    ),
):
    """Runs pricing and performance benchmarks for the library's models."""
    all_configs = get_benchmark_configs()
    configs_to_run = all_configs

    if model:
        model_name_map = {
            cfg["model_name"].split(" ")[0].upper(): cfg for cfg in all_configs
        }
        if model.upper() in model_name_map:
            configs_to_run = [model_name_map[model.upper()]]
        else:
            typer.secho(f"Error: Model '{model}' not found.", fg=typer.colors.RED)
            valid_names = ", ".join(
                [cfg["model_name"].split(" ")[0] for cfg in all_configs]
            )
            console.print(f"Available models: [bold]{valid_names}[/bold]")
            raise typer.Exit(code=1)

    if technique:
        for cfg in configs_to_run:
            current_techs = cfg["techniques"]
            # Create a case-insensitive map for matching
            tech_name_map = {
                key.upper().split(" ")[0]: (key, val)
                for key, val in current_techs.items()
            }
            if technique.upper() in tech_name_map:
                key, val = tech_name_map[technique.upper()]
                cfg["techniques"] = {key: val}
            else:
                cfg["techniques"] = {}

    for config in configs_to_run:
        if config["techniques"]:
            run_benchmark(config)


if __name__ == "__main__":
    app()
