import sys
import time
from benchmark.runner import run_from_config

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

# A list of just the demo files.
DEMO_BENCHMARKS = [
    "benchmark.configs.demo_blacks_approx",
    "benchmark.configs.demo_perpetual_put",
    "benchmark.configs.demo_parity_tools",
]

def main(benchmarks_to_run: list[str]):
    """
    Runs a list of specified benchmarks.
    """
    total_start_time = time.time()
    print(f"--- Starting Master Benchmark Runner ---")
    print(f"Found {len(benchmarks_to_run)} benchmarks to run.")
    
    for i, config_path in enumerate(benchmarks_to_run):
        print("\n" + "#" * 80)
        print(f"### RUNNING BENCHMARK {i+1}/{len(benchmarks_to_run)}: {config_path}")
        print("#" * 80)
        
        start_time = time.time()
        try:
            run_from_config(config_path)
        except Exception as e:
            print(f"\n!!!!!! ERROR running {config_path} !!!!!!")
            print(f"Error type: {type(e).__name__}")
            print(f"Error details: {e}")
            # You might want to print the full traceback here for debugging
            import traceback
            traceback.print_exc()
        
        end_time = time.time()
        print(f"--- Finished {config_path} in {end_time - start_time:.2f} seconds ---")

    total_end_time = time.time()
    print("\n" + "=" * 80)
    print(f"Master Benchmark Runner Finished. Total time: {total_end_time - total_start_time:.2f} seconds.")
    print("=" * 80)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # provided specific benchmarks to run
        benchmarks = sys.argv[1:]
    else:
        # Default: run all standard benchmarks
        benchmarks = ALL_BENCHMARKS
        
    main(benchmarks)