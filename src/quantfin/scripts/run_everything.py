"""
run_everything.py

Quick smoke-test for your newly packaged quantfin library:
- Full benchmark suite
- Consolidated benchmark report
- Selected benchmark run
- Daily calibration (on SPY)
- Backtest (on SPY with BSM & Merton)
"""

import json

from quantfin.benchmark.run_all_benchmarks          import ALL_BENCHMARKS, main as full
from quantfin.benchmark.run_consolidated_benchmarks import main as prices
from quantfin.workflows.run_calib                   import main as main_calib
from quantfin.workflows.run_backtest                import main as run_backtest
from quantfin.plotting.run_visualization            import main as graph  # if you want to test visualization

def main(benchmarks_to_run: list[str] | None = None):
    if benchmarks_to_run is None:
        benchmarks_to_run = ALL_BENCHMARKS
    print("\n>>> Running full benchmark suite\n")
    full(ALL_BENCHMARKS)

    print("\n>>> Running consolidated benchmark report\n")
    prices(ALL_BENCHMARKS)

    print("\n>>> Running a selected benchmark (demo_bsm) only\n")
    slct = ["quantfin.benchmark.configs.demo_bsm"]
    #full(slct)
    #prices(slct)

    print("\n>>> Running daily calibration for SPY\n")
    main_calib("SPY")

    print("\n>>> Running backtest for SPY with BSM & Merton\n")
    run_backtest("SPY", ["BSM", "Merton"])

    # If you want to test your visualization entry-point, uncomment and provide a valid date:
    # print("\n>>> Generating visualization for SPY on 2025-06-18\n")
    # graph("SPY", "2025-06-18")

if __name__ == "__main__":
    main()
