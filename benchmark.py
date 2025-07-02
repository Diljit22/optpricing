import math
import time
from typing import Any

import numpy as np
from tabulate import tabulate

from optpricing.atoms import Option, OptionType, Rate, Stock, ZeroCouponBond
from optpricing.models import (
    BatesModel,
    BSMModel,
    CEVModel,
    CGMYModel,
    CIRModel,
    HestonModel,
    HyperbolicModel,
    KouModel,
    MertonJumpModel,
    NIGModel,
    SABRJumpModel,
    SABRModel,
    VarianceGammaModel,
    VasicekModel,
)
from optpricing.models.base.base_model import BaseModel
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


def profile_all_metrics(
    technique: Any, option: Option, stock: Stock, model: BaseModel, rate: Rate, **kwargs
):
    """Helper to run and profile all calculations for a given technique."""
    results = {}

    # Price
    start = time.perf_counter()
    try:
        price_val = technique.price(option, stock, model, rate, **kwargs).price
    except (NotImplementedError, TypeError, ValueError):
        price_val = np.nan
    end = time.perf_counter()
    results["Price"] = (price_val, end - start)

    # Greeks
    greeks_to_calc = {
        "Delta": technique.delta,
        "Gamma": technique.gamma,
        "Vega": technique.vega,
        "Theta": technique.theta,
        "Rho": technique.rho,
    }
    for name, func in greeks_to_calc.items():
        start = time.perf_counter()
        try:
            # For cached greeks, this will be fast. For others, it reruns.
            greek_val = func(option, stock, model, rate, **kwargs)
        except (NotImplementedError, TypeError, ValueError):
            greek_val = np.nan
        end = time.perf_counter()
        results[name] = (greek_val, end - start)

    # Implied Volatility
    start = time.perf_counter()
    try:
        if np.isfinite(price_val):
            iv_val = technique.implied_volatility(
                option, stock, model, rate, target_price=price_val, **kwargs
            )
        else:
            iv_val = np.nan
    except (NotImplementedError, TypeError, ValueError, RuntimeError):
        iv_val = np.nan
    end = time.perf_counter()
    results["ImpliedVol"] = (iv_val, end - start)

    return results


def run_benchmark(config: dict[str, Any]):
    """Runs and prints a formatted benchmark for a given model configuration."""
    model = config["model_instance"]
    stock = config["stock"]
    rate = config["rate"]
    techniques = config["techniques"]
    kwargs = config.get("kwargs", {})

    print("\n" + "=" * 80)
    print(f" {config['model_name']} ".center(80, "="))
    print("=" * 80)
    print(f"Model: {model}")
    print(f"Stock: {stock}")
    print(f"Rate: {rate}")

    all_results: dict[str, dict[str, Any]] = {}

    for option_type in [OptionType.CALL, OptionType.PUT]:
        option = Option(strike=100.0, maturity=1.0, option_type=option_type)

        # Special handling for Dupire strikes
        if model.name == "Dupire Local Volatility":
            option = config["options"][option_type]

        header = (
            f" OPTION: {option.option_type.value.upper()} | "
            f"K={option.strike} | T={option.maturity} "
        ).center(80, "-")

        print(f"\n{header}")

        tech_results = {}
        for name, tech_instance in techniques.items():
            tech_results[name] = profile_all_metrics(
                tech_instance, option, stock, model, rate, **kwargs
            )

        all_results[option_type.value] = tech_results

        # Format and print the results table
        headers = ["Metric"] + list(techniques.keys())
        metrics = ["Price", "Delta", "Gamma", "Vega", "Theta", "Rho", "ImpliedVol"]
        table_data = []
        for metric in metrics:
            row = [metric]
            for tech_name in techniques.keys():
                value, timing_s = tech_results[tech_name][metric]
                row.append(f"{value:.4f}\n({timing_s:.4f} s)")
            table_data.append(row)
        print(tabulate(table_data, headers=headers, tablefmt="grid", stralign="center"))

    # Put-Call Parity Error
    if (
        model.name != "Dupire Local Volatility"
    ):  # Parity doesn't apply if strikes differ
        print("\n" + " Put-Call Parity Errors: ".center(80, "-"))
        parity_errors = {}
        S, K, T = stock.spot, option.strike, option.maturity
        r, q = rate.get_rate(T), stock.dividend
        expected_diff = S * math.exp(-q * T) - K * math.exp(-r * T)

        for tech_name in techniques.keys():
            call_price = all_results["Call"][tech_name]["Price"][0]
            put_price = all_results["Put"][tech_name]["Price"][0]
            if np.isfinite(call_price) and np.isfinite(put_price):
                parity_errors[tech_name] = call_price - put_price - expected_diff
            else:
                parity_errors[tech_name] = np.nan

        headers = list(parity_errors.keys())
        error_data = [[f"{err:.4e}" for err in parity_errors.values()]]
        print(
            tabulate(error_data, headers=headers, tablefmt="plain", stralign="center")
        )


# Main
if __name__ == "__main__":
    stock = Stock(spot=100.0, dividend=0.01)
    rate = Rate(rate=0.03)

    benchmark_configs = [
        {
            "model_name": "BLACK-SCHOLES-MERTON MODEL VS. ALL TECHNIQUES",
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
                "MC": MonteCarloTechnique(n_paths=50000, n_steps=100, seed=42),
            },
        },
        {
            "model_name": "MERTON JUMP-DIFFUSION MODEL",
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
                "MC": MonteCarloTechnique(n_paths=50000, n_steps=100, seed=42),
            },
        },
        {
            "model_name": "HESTON STOCHASTIC VOLATILITY MODEL",
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
                "MC": MonteCarloTechnique(n_paths=50000, n_steps=100, seed=42),
            },
        },
        {
            "model_name": "BATES (HESTON + JUMPS) MODEL",
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
                "MC": MonteCarloTechnique(n_paths=50000, n_steps=100, seed=42),
            },
        },
        {
            "model_name": "VARIANCE GAMMA (VG) MODEL",
            "model_instance": VarianceGammaModel(
                params={"sigma": 0.2, "nu": 0.1, "theta": -0.14}
            ),
            "stock": stock,
            "rate": rate,
            "techniques": {
                "Integration": IntegrationTechnique(),
                "FFT": FFTTechnique(),
                "MC": MonteCarloTechnique(n_paths=50000, seed=42),
            },
        },
        {
            "model_name": "KOU DOUBLE-EXPONENTIAL JUMP MODEL",
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
                "MC": MonteCarloTechnique(n_paths=50000, n_steps=100, seed=42),
            },
        },
        {
            "model_name": "CGMY MODEL",
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
            "model_name": "NORMAL INVERSE GAUSSIAN (NIG) MODEL",
            "model_instance": NIGModel(
                params={"alpha": 15.0, "beta": -5.0, "delta": 0.5}
            ),
            "stock": stock,
            "rate": rate,
            "techniques": {
                "Integration": IntegrationTechnique(),
                "FFT": FFTTechnique(),
                "MC": MonteCarloTechnique(n_paths=50000, seed=42),
            },
        },
        {
            "model_name": "HYPERBOLIC MODEL",
            "model_instance": HyperbolicModel(
                params={"alpha": 15.0, "beta": -5.0, "delta": 0.5, "mu": 0.0}
            ),
            "stock": stock,
            "rate": rate,
            "techniques": {
                "Integration": IntegrationTechnique(),
                "FFT": FFTTechnique(),
                # "MC": MonteCarloTechnique(n_paths=50000, seed=42),
            },
        },
        {
            "model_name": "CONSTANT ELASTICITY OF VARIANCE (CEV) MODEL",
            "model_instance": CEVModel(params={"sigma": 0.8, "gamma": 0.7}),
            "stock": stock,
            "rate": rate,
            "techniques": {
                "MC (Exact)": MonteCarloTechnique(n_paths=50000, seed=42),
            },
        },
        {
            "model_name": "SABR MODEL",
            "model_instance": SABRModel(
                params={"alpha": 0.5, "beta": 0.8, "rho": -0.6}
            ),
            "stock": stock,
            "rate": rate,
            "kwargs": {"v0": 0.5},
            "techniques": {
                "MC": MonteCarloTechnique(n_paths=50000, n_steps=100, seed=42),
            },
        },
        {
            "model_name": "SABR JUMP MODEL",
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
            "techniques": {
                "MC": MonteCarloTechnique(n_paths=50000, n_steps=100, seed=42),
            },
        },
    ]

    # Run all benchmarks
    for config in benchmark_configs:
        run_benchmark(config)

    # Test Interest Rate Models
    print("\n" + "=" * 80)
    print("--- TESTING INTEREST RATE MODELS ---".center(80))
    print("=" * 80)

    # same technique, different model and "asset"
    cf_technique = ClosedFormTechnique()

    # Define a Zero-Coupon Bond to be priced
    bond = ZeroCouponBond(maturity=1.0, face_value=1.0)

    # Pass initial short rate r0 via the stock.spot
    initial_short_rate = 0.05
    r0_stock = Stock(spot=initial_short_rate)

    # The Rate object is required by the technique, will be ignored by the models
    dummy_rate = Rate(rate=0.99)

    # Vasicek Model Test
    vasicek_model = VasicekModel(params={"kappa": 0.86, "theta": 0.09, "sigma": 0.02})
    vasicek_price = cf_technique.price(bond, r0_stock, vasicek_model, dummy_rate).price

    print(
        f"Vasicek ZCB Price (r0={initial_short_rate:.2f}, "
        f"T={bond.maturity:.1f}): {vasicek_price:.6f}"
    )

    # CIR Model Test
    cir_model = CIRModel(params={"kappa": 0.86, "theta": 0.09, "sigma": 0.02})
    cir_price = cf_technique.price(bond, r0_stock, cir_model, dummy_rate).price

    print(
        f"CIR ZCB Price     (r0={initial_short_rate:.2f}, "
        f"T={bond.maturity:.1f}): {cir_price:.6f}"
    )
