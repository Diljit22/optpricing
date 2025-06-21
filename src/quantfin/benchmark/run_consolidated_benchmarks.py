import importlib
import sys
import traceback

from quantfin.benchmark.consolidated_report import print_consolidated_report
from quantfin.benchmark.runner import collect_benchmark_data

# A list of all benchmark configurations you want to run.
ALL_BENCHMARKS = [
    "quantfin.benchmark.configs.demo_bsm",
    "quantfin.benchmark.configs.demo_merton_jump",
    "quantfin.benchmark.configs.demo_heston",
    "quantfin.benchmark.configs.demo_bates",
    "quantfin.benchmark.configs.demo_vg",
    "quantfin.benchmark.configs.demo_kou",
    "quantfin.benchmark.configs.demo_cgmy",
    "quantfin.benchmark.configs.demo_nig",
    "quantfin.benchmark.configs.demo_hyperbolic",
    "quantfin.benchmark.configs.demo_cev",
    "quantfin.benchmark.configs.demo_sabr",
    "quantfin.benchmark.configs.demo_sabr_jump",
    "quantfin.benchmark.configs.demo_dupire",
]

# A list of just the demo files.
DEMO_BENCHMARKS = [
    "quantfin.benchmark.configs.demo_blacks_approx",
    "quantfin.benchmark.configs.demo_perpetual_put",
    "quantfin.benchmark.configs.demo_parity_tools",
]



def main(benchmarks_to_run: list[str]):
    """
    Loads a list of benchmark configs, executes them to collect price data,
    and then prints a single consolidated report.
    """
    print("--- Starting Consolidated Benchmark Runner ---")
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
