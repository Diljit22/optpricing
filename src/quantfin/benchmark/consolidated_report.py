from __future__ import annotations
import traceback
from typing import List, Dict, Any
from quantfin.atoms.option import OptionType

def print_consolidated_report(all_run_results: List[Dict[str, Any]]) -> None:
    """
    Prints a consolidated summary report from a list of benchmark run results.
    """
    print("\n" + "=" * 80)
    print("CONSOLIDATED BENCHMARK RESULTS")
    print("=" * 80)

    for run_result in all_run_results:
        print(f"\n{run_result['name'].upper()}")
        print("-" * 80)

        # --- Handle and Print Errors ---
        if run_result.get("error"):
            print(f"[ERROR] Benchmark failed: {run_result['error']}")
            # Optionally print the full traceback for debugging
            print("\n--- Traceback ---")
            print(run_result['traceback'])
            print("-----------------\n")
            continue

        # --- If no error, print the results table ---
        techniques = run_result['techniques']
        results = run_result['results']
        timings = run_result['timings']
        options = run_result['options']
        
        col_width = 14
        header = f"{'Metric':<12}" + "".join([f"{name:^{col_width}}" for name in techniques])
        print(header)
        print("-" * len(header))

        # Find call and put options safely
        call_opt = next((o for o in options if o.option_type == OptionType.CALL), None)
        put_opt = next((o for o in options if o.option_type == OptionType.PUT), None)

        # Print Call Row
        if call_opt:
            row_val = f"{'Call':<12}"
            row_time = f"{'(seconds)':<12}"
            for tech_name in techniques:
                val = results[tech_name][call_opt]['Price']
                time = timings[tech_name][call_opt]['Price']
                row_val += f"{val:^{col_width}.4f}"
                row_time += f"{time:^{col_width}.4f}"
            print(row_val)
            print(row_time)

        # Print Put Row
        if put_opt:
            row_val = f"{'Put':<12}"
            row_time = f"{'(seconds)':<12}"
            for tech_name in techniques:
                val = results[tech_name][put_opt]['Price']
                time = timings[tech_name][put_opt]['Price']
                row_val += f"{val:^{col_width}.4f}"
                row_time += f"{time:^{col_width}.4f}"
            print(row_val)
            print(row_time)

    print("\n" + "=" * 80)
    print("Consolidated Run Finished.")
    print("=" * 80)