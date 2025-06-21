
import pandas as pd

from quantfin.config import BACKTEST_LOGS_DIR
from quantfin.data import get_available_snapshot_dates, load_market_snapshot

from .daily_workflow import DailyWorkflow


class BacktestWorkflow:
    """
    Orchestrates a backtest for a single model over multiple historical snapshots.
    """
    def __init__(self, ticker: str, model_config: dict):
        self.ticker = ticker.upper()
        self.model_config = model_config
        self.results = []

    def run(self):
        available_dates = get_available_snapshot_dates(self.ticker)
        if len(available_dates) < 2:
            print(f"Backtest for {self.model_config['name']} requires at least 2 days of data. Skipping.")
            return

        for i in range(len(available_dates) - 1):
            calib_date, eval_date = available_dates[i], available_dates[i+1]
            print(f"\n--- Processing Period: Calibrate on {calib_date}, Evaluate on {eval_date} ---")

            calib_data = load_market_snapshot(self.ticker, calib_date)
            eval_data = load_market_snapshot(self.ticker, eval_date)
            if calib_data is None or eval_data is None: continue

            # Run a daily workflow to get the calibrated model
            calib_workflow = DailyWorkflow(market_data=calib_data, model_config=self.model_config)
            calib_workflow.run()

            if calib_workflow.results['Status'] != 'Success':
                print("    -> Calibration failed. Skipping evaluation for this period.")
                continue

            calibrated_model = self.model_config['model_class'](params=calib_workflow.results['Calibrated Params'])

            # Evaluate the model on the next day's data
            eval_workflow = DailyWorkflow(market_data=eval_data, model_config=self.model_config)
            # need to run the first part of the eval workflow to get the correct r, q, and stock
            eval_workflow.run()
            if eval_workflow.results['Status'] != 'Success':
                print("    -> Evaluation setup failed. Skipping evaluation for this period.")
                continue

            rmse = eval_workflow._evaluate_rmse(calibrated_model, eval_workflow.stock, eval_workflow.rate)

            print(f"  -> Out-of-Sample RMSE for {self.model_config['name']} on {eval_date}: {rmse:.4f}")
            self.results.append({'Eval Date': eval_date, 'Model': self.model_config['name'], 'Out-of-Sample RMSE': rmse})

    def save_results(self):
        """Saves the collected backtest results to a CSV file."""
        if not self.results:
            print("No backtest results to save.")
            return

        df = pd.DataFrame(self.results)
        today_str = pd.Timestamp.now().strftime('%Y-%m-%d')
        filepath = BACKTEST_LOGS_DIR / f"{self.ticker}_backtest_{today_str}.csv"
        df.to_csv(filepath, index=False)
        print(f"\nDetailed backtest log saved to: {filepath}")
