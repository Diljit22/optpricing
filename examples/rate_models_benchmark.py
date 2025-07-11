from __future__ import annotations

import typer
from rich.console import Console

from optpricing.atoms import Rate, Stock, ZeroCouponBond
from optpricing.models import CIRModel, VasicekModel
from optpricing.techniques import ClosedFormTechnique

__doc__ = """
A benchmark for pricing Zero-Coupon Bonds using various short-rate models.
"""

app = typer.Typer(
    name="rate-benchmark",
    help="Runs benchmark demos for interest rate models.",
    add_completion=False,
    no_args_is_help=True,
)
console = Console()


@app.command()
def main():
    """
    Prices a Zero-Coupon Bond using the Vasicek and CIR models and prints the results.
    """
    console.rule("[bold cyan]INTEREST RATE MODELS[/bold cyan]", style="cyan")
    bond = ZeroCouponBond(maturity=1.0, face_value=1.0)
    # For rate models, stock.spot is interpreted as the initial short rate r0
    r0_stock = Stock(spot=0.05)
    # The rate atom is ignored by rate models but required by the technique signature
    dummy_rate = Rate(rate=0.0)
    cf_technique = ClosedFormTechnique()

    # Vasicek Model
    vasicek_model = VasicekModel(params={"kappa": 0.86, "theta": 0.09, "sigma": 0.02})
    vasicek_price = cf_technique.price(bond, r0_stock, vasicek_model, dummy_rate).price
    console.print(f"Vasicek Model: {vasicek_model}")
    console.print(
        f" -> ZCB Price (r0=0.05, T=1.0): "
        f"[bold green]{vasicek_price:.6f}[/bold green]\n"
    )

    # CIR Model
    cir_model = CIRModel(params={"kappa": 0.86, "theta": 0.09, "sigma": 0.02})
    cir_price = cf_technique.price(bond, r0_stock, cir_model, dummy_rate).price
    console.print(f"CIR Model: {cir_model}")
    console.print(
        f" -> ZCB Price (r0=0.05, T=1.0): [bold green]{cir_price:.6f}[/bold green]"
    )


if __name__ == "__main__":
    app()
