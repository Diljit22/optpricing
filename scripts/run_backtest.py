import pandas as pd

from quantfin.workflows import BacktestWorkflow
from quantfin.workflows.configs import BSM_WORKFLOW_CONFIG, MERTON_WORKFLOW_CONFIG

MODELS_TO_BACKTEST = [
    BSM_WORKFLOW_CONFIG,
    MERTON_WORKFLOW_CONFIG,
]

def main(ticker: str):
    """Executes backtests for specified models on a ticker."""
    all_results = []
    for model_config in MODELS_TO_BACKTEST:
        backtester = BacktestWorkflow(ticker=ticker, model_config=model_config)
        backtester.run()
        all_results.extend(backtester.results)

    if not all_results:
        print("No results generated from backtest.")
        return

    results_df = pd.DataFrame(all_results)
    pivot_df = results_df.pivot(index='Eval Date', columns='Model', values='Out-of-Sample RMSE')

    print("\n\n" + "="*80)
    print(f"--- FINAL BACKTEST SUMMARY for {ticker} ---")
    print("="*80)
    print("Out-of-Sample RMSE by Day:")
    print(pivot_df.to_string(float_format="%.4f"))
    print("\nAverage RMSE per Model:")
    print(pivot_df.mean().to_string(float_format="%.4f"))

    dummy_backtester = BacktestWorkflow(ticker, {})
    dummy_backtester.results = all_results
    dummy_backtester.save_results()

if __name__ == '__main__':
    TICKER = 'SPY'
    main(TICKER)
