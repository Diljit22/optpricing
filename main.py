from benchmark.run_all_benchmarks import ALL_BENCHMARKS, main as full
from benchmark.run_consolidated_benchmarks import main as prices
from workflows.run_calib import main as main_calib
from workflows.run_backtest import main as run_backtest
from plotting.run_visualization import main as graph

#full(ALL_BENCHMARKS)
prices(ALL_BENCHMARKS)

#slct = ["benchmark.configs.demo_bsm",]
#full(slct)
#prices(slct)

#main_calib('SPY')
#run_backtest('SPY', ['BSM', 'Merton'])
