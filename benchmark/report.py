from __future__ import annotations
import math
from typing import Any, Sequence

from atoms.option import OptionType

def print_benchmark_report(
    config: Any,
    all_results: Sequence[dict[Any, dict[str, float]]],
    all_timings: Sequence[dict[Any, dict[str, float]]],
    parity_threshold: float = 0.005,
) -> None:
    """
    Prints a comparison of pricing techniques. If both a call and put are present
    in the config, it also includes per-technique put-call parity errors.
    """
    # --- Header ---
    print("=" * 80)
    print(f"{config.name.upper()}") # Use the config name for a better title
    print("=" * 80)
    print(f"Model: {config.model}")
    print(f"Stock: {config.stock}")
    print(f"Rate: {config.rate}")
    print("-" * 80)

    technique_names = [name for _, name in config.techniques]
    col_width = 14

    # --- Check if a parity check is possible ---
    has_call = any(o.option_type == OptionType.CALL for o in config.options)
    has_put = any(o.option_type == OptionType.PUT for o in config.options)
    parity_is_possible = has_call and has_put

    if parity_is_possible:
        call_opt = next(o for o in config.options if o.option_type == OptionType.CALL)
        put_opt  = next(o for o in config.options if o.option_type == OptionType.PUT)
        # Precompute discount differential for parity check
        S0, q, r = config.stock.spot, config.stock.dividend, config.rate.rate
        T = call_opt.maturity # Assume same maturity for the pair
        diff = S0 * math.exp(-q * T) - call_opt.strike * math.exp(-r * T)

    # --- Main Loop for Each Option Priced ---
    for opt in config.options:
        opt_label = opt.option_type.name # Use .name for "CALL" or "PUT"

        print(f"\n{' ' * 20}{'-' * 40}")
        print(f"{' ' * 23}OPTION: {opt_label} | K={opt.strike:.1f} | T={opt.maturity:.3f}")
        print(f"{' ' * 20}{'-' * 40}\n")

        metrics = config.metrics_to_run

        # Print metric and timing table
        header = f"{'Metric':<12}" + "".join([f"{name:^{col_width}}" for name in technique_names])
        print(header)
        print("-" * len(header))

        for metric in metrics:
            row_val = f"{metric:<12}"
            row_time = f"{'(seconds)':<12}"
            for res, times in zip(all_results, all_timings):
                val = res[opt].get(metric, float('nan')) # Use .get for safety
                t   = times[opt].get(metric, 0.0)
                row_val  += f"{val:^{col_width}.4f}"
                row_time += f"{t:^{col_width}.4f}"
            print(row_val)
            print(row_time)
            if metric == "Price": print() # Add space after price

    # --- Parity Check Section (if possible) ---
    if parity_is_possible:
        print(f"\n{' ' * 20}{'-' * 40}")
        print(f"{' ' * 23}Put-Call Parity Errors:")
        print(f"{' ' * 20}{'-' * 40}\n")

        parity_header = f"{'Technique':<12}" + "".join([f"{name:^{col_width}}" for name in technique_names])
        print(parity_header)
        print("-" * len(parity_header))

        err_row = f"{'Error':<12}"
        for res in all_results:
            call_price = res[call_opt]['Price']
            put_price  = res[put_opt]['Price']
            err = call_price - put_price - diff
            marker = '*' if abs(err) > parity_threshold else ' '
            err_row += f"{err:^{col_width}.4e}{marker}" # Use scientific notation for small errors
        print(err_row)
        print()
        print(f"* indicates |error| > {parity_threshold}\n")

    print("Benchmark complete.")
    