import pandas as pd
import os
from datetime import date

from workflows.daily_workflow import DailyWorkflow
from data.data_manager import get_available_snapshot_dates, load_market_snapshot
from data.historical_manager import load_historical_returns
from calibration.fit_jump_parameters import fit_jump_params_from_history

from workflows.configs.bsm_config import BSM_WORKFLOW_CONFIG
from workflows.configs.heston_config import HESTON_WORKFLOW_CONFIG
from workflows.configs.merton_config import MERTON_WORKFLOW_CONFIG
from workflows.configs.kou_config import KOU_WORKFLOW_CONFIG
from workflows.configs.vg_config import VG_WORKFLOW_CONFIG
from workflows.configs.nig_config import NIG_WORKFLOW_CONFIG

WORKFLOWS_TO_RUN = [
    BSM_WORKFLOW_CONFIG,
    MERTON_WORKFLOW_CONFIG,
    #KOU_WORKFLOW_CONFIG,
]

def main(ticker: str, snapshot_date: str=None):
    if snapshot_date is None: snapshot_date = get_available_snapshot_dates(ticker)[-1]

    print(f"--- Starting Evaluation for {ticker} on {snapshot_date} ---")
    market_data = load_market_snapshot(ticker, snapshot_date)
    if market_data is None: return

    print("\n--- Pre-computing historical parameters ---")
    log_returns = load_historical_returns(ticker)
    historical_params = fit_jump_params_from_history(log_returns)
    
    all_results = []
    for model_config in WORKFLOWS_TO_RUN:
        workflow = DailyWorkflow(
            market_data=market_data,
            model_config=model_config,
            historical_params=historical_params
        )
        workflow.run()
        all_results.append(workflow.results)
    
    print("\n\n" + "="*80)
    print(f"--- FINAL SUMMARY: {ticker} on {snapshot_date} ---")
    print("="*80)
    report_df = pd.DataFrame(all_results)
    print(report_df.to_string(index=False, float_format="%.4f"))

if __name__ == '__main__':
    TICKER = 'SPY'
    latest_date = get_available_snapshot_dates(TICKER)[-1]
    main(TICKER, latest_date)