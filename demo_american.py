from __future__ import annotations

import time

import typer
from rich.console import Console
from rich.table import Table

from optpricing.atoms import Option, OptionType, Rate, Stock
from optpricing.models import (
    BatesModel,
    BSMModel,
    HestonModel,
    KouModel,
    MertonJumpModel,
    SABRJumpModel,
    SABRModel,
)
from optpricing.techniques import CRRTechnique, LeisenReimerTechnique, TOPMTechnique
from optpricing.techniques.american_monte_carlo import AmericanMonteCarloTechnique

console = Console()
app = typer.Typer(no_args_is_help=True)


def run_american_benchmark(config: dict):
    """Runs and prints a formatted benchmark for a given model configuration."""
    model = config["model_instance"]
    stock = config["stock"]
    rate = config["rate"]
    techniques = config["techniques"]
    kwargs = config.get("kwargs", {})

    console.rule(f"[bold cyan]{config['model_name']}[/bold cyan]", style="cyan")
    console.print(f"[bold]Model:[/] {model}")

    option = Option(strike=110.0, maturity=1.0, option_type=OptionType.PUT)
    console.print("[bold]Option:[/] American Put, K=110, T=1.0")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Technique", style="dim", width=25)
    table.add_column("Price", justify="center")
    table.add_column("Time (s)", justify="center")

    for name, tech_instance in techniques.items():
        start = time.perf_counter()
        try:
            price = tech_instance.price(option, stock, model, rate, **kwargs).price
        except (NotImplementedError, TypeError, ValueError) as e:
            price = f"[bold red]Error: {e}[/bold red]"

        end = time.perf_counter()
        timing_s = end - start

        if isinstance(price, float):
            table.add_row(name, f"{price:.4f}", f"{timing_s:.4f}")
        else:
            table.add_row(name, price, f"{timing_s:.4f}")

    console.print(table)


@app.command()
def main():
    """
    Runs a comparison of American option pricing techniques for various models.
    """
    stock = Stock(spot=100.0, dividend=0.01)
    rate = Rate(rate=0.05)

    # Define the American MC technique we want to test
    american_mc = AmericanMonteCarloTechnique(n_paths=20000, n_steps=100, seed=42)

    benchmark_configs = [
        {
            "model_name": "BLACK-SCHOLES-MERTON",
            "model_instance": BSMModel(params={"sigma": 0.2}),
            "stock": stock,
            "rate": rate,
            "techniques": {
                "American MC (LSMC)": american_mc,
                "Binomial Tree (CRR)": CRRTechnique(
                    steps=501,
                    is_american=True,
                ),
                "Leisen-Reimer Tree": LeisenReimerTechnique(
                    steps=501,
                    is_american=True,
                ),
                "Trinomial Tree (TOPM)": TOPMTechnique(steps=501, is_american=True),
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
            "techniques": {"American MC (LSMC)": american_mc},
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
            "techniques": {"American MC (LSMC)": american_mc},
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
            "techniques": {"American MC (LSMC)": american_mc},
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
            "techniques": {"American MC (LSMC)": american_mc},
        },
        {
            "model_name": "SABR",
            "model_instance": SABRModel(
                params={"alpha": 0.5, "beta": 0.8, "rho": -0.6}
            ),
            "stock": stock,
            "rate": rate,
            "kwargs": {"v0": 0.5},
            "techniques": {"American MC (LSMC)": american_mc},
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
            "techniques": {"American MC (LSMC)": american_mc},
        },
    ]

    for config in benchmark_configs:
        run_american_benchmark(config)


if __name__ == "__main__":
    app()
