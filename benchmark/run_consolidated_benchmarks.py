import importlib
import sys
import time
import traceback

from benchmark.runner import collect_benchmark_data
from benchmark.consolidated_report import print_consolidated_report

# A list of all benchmark configurations you want to run.
ALL_BENCHMARKS = [
    "benchmark.configs.demo_bsm",
    "benchmark.configs.demo_merton_jump",
    "benchmark.configs.demo_heston",
    "benchmark.configs.demo_bates",
    "benchmark.configs.demo_vg",
    "benchmark.configs.demo_kou",
    "benchmark.configs.demo_cgmy",
    "benchmark.configs.demo_nig",
    "benchmark.configs.demo_hyperbolic",
    "benchmark.configs.demo_cev",
    "benchmark.configs.demo_sabr",
    "benchmark.configs.demo_sabr_jump",
    "benchmark.configs.demo_dupire",    
]


def main(benchmarks_to_run: list[str]):
    """
    Loads a list of benchmark configs, executes them to collect price data,
    and then prints a single consolidated report.
    """
    print(f"--- Starting Consolidated Benchmark Runner ---")
    print(f"Found {len(benchmarks_to_run)} benchmarks to run.")
    
    all_run_results = []

    for config_path in benchmarks_to_run:
        print("\n" + "#" * 80)
        print(f"### EXECUTING: {config_path}")
        print("#" * 80)
        
        try:
            # Load the config module
            config_module = importlib.import_module(config_path)
            config = config_module.config

            # Run the collector to get data
            results, timings = collect_benchmark_data(config)

            # Store results for the final report
            all_run_results.append({
                "name": config.name,
                "techniques": [name for _, name in config.techniques],
                "results": results,
                "timings": timings,
                "options": config.options,
                "error": None,
                "traceback": None,
            })

        except Exception as e:
            print(f"\n!!!!!! ERROR executing {config_path} !!!!!!")
            # Store the error information for the final report
            all_run_results.append({
                "name": f"{config_path} (FAILED)",
                "error": e,
                "traceback": traceback.format_exc(),
            })
        
        print(f"--- Finished executing {config_path} ---")

    # After all benchmarks have been run (or failed), print the single report
    print_consolidated_report(all_run_results)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        benchmarks = sys.argv[1:]
    else:
        benchmarks = ALL_BENCHMARKS
        
    main(benchmarks)