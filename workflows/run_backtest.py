import pandas as pd
import os
from datetime import date

from workflows.backtest_workflow import BacktestWorkflow

from data.historical_manager import load_historical_returns
from calibration.fit_jump_parameters import fit_jump_params_from_history

from models.bsm import BSMModel
from models.merton_jump import MertonJumpModel
from workflows.configs.bsm_config import BSM_WORKFLOW_CONFIG
from workflows.configs.merton_config import MERTON_WORKFLOW_CONFIG

MODELS_TO_BACKTEST = {
    "BSM": BSM_WORKFLOW_CONFIG,
    "Merton": MERTON_WORKFLOW_CONFIG,
}

def main(ticker: str, models_to_run: list[str]):
    """
    Main function to execute backtests for specified models on a ticker.
    """
    print("\n" + "="*80)
    print(f"--- Starting Backtest Suite for Ticker: {ticker} ---")
    print("="*80)

    # 1. Pre-compute historical parameters once
    historical_params = fit_jump_params_from_history(load_historical_returns(ticker))
    
    all_results = []

    # 2. Loop through each model to test
    for model_name in models_to_run:
        if model_name not in MODELS_TO_BACKTEST:
            print(f"Warning: Model '{model_name}' not found in configuration. Skipping.")
            continue
            
        # Create and run the backtester for this model
        backtester = BacktestWorkflow(
            ticker=ticker,
            model_config=MODELS_TO_BACKTEST[model_name],
            historical_params=historical_params
        )
        backtester.run()
        all_results.extend(backtester.results)

    # 3. Report Final Results
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

    # Save the log file
    log_dir = "backtest_logs"
    os.makedirs(log_dir, exist_ok=True)
    filename = os.path.join(log_dir, f"{ticker}_backtest_{date.today()}.csv")
    results_df.to_csv(filename, index=False)
    print(f"\nDetailed backtest log saved to: {filename}")