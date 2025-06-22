import pandas as pd

from quantfin.data import get_available_snapshot_dates, load_market_snapshot
from quantfin.workflows import DailyWorkflow
from quantfin.workflows.configs import (
    BSM_WORKFLOW_CONFIG,
    HESTON_WORKFLOW_CONFIG,
    MERTON_WORKFLOW_CONFIG,
)

WORKFLOWS_TO_RUN = [
    BSM_WORKFLOW_CONFIG,
    MERTON_WORKFLOW_CONFIG,
    HESTON_WORKFLOW_CONFIG,
]


def main(ticker: str, snapshot_date: str = None):
    """Runs the daily calibration workflow for a set of models."""
    if snapshot_date is None:
        snapshot_date = get_available_snapshot_dates(ticker)[0]  # Get most recent

    print(f"--- Starting Evaluation for {ticker} on {snapshot_date} ---")
    market_data = load_market_snapshot(ticker, snapshot_date)
    if market_data is None:
        return

    all_results = []
    for model_config in WORKFLOWS_TO_RUN:
        # Each model gets its own workflow instance
        workflow = DailyWorkflow(market_data=market_data, model_config=model_config)
        workflow.run()
        all_results.append(workflow.results)

    print("\n\n" + "=" * 80)
    print(f"--- FINAL SUMMARY: {ticker} on {snapshot_date} ---")
    print("=" * 80)
    report_df = pd.DataFrame(all_results)
    print(report_df.to_string(index=False, float_format="%.4f"))


if __name__ == "__main__":
    TICKER = "SPY"
    main(TICKER)
