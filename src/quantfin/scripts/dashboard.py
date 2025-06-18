import streamlit as st
import pandas as pd
import os
import json
from datetime import date

from quantfin.data.data_manager               import (
    load_market_snapshot,
    get_available_snapshot_dates,
)
from quantfin.workflows.daily_workflow        import DailyWorkflow
from quantfin.calibration.iv_surface          import VolatilitySurface
from quantfin.plotting.smile_plotter          import plot_smiles_by_expiry
from quantfin.atoms.stock                     import Stock
from quantfin.atoms.rate                      import Rate
from quantfin.calibration.fit_market_params   import fit_rate_and_dividend
from quantfin.calibration.technique_selector  import select_fastest_technique
from quantfin.atoms.option                    import Option, OptionType

from quantfin.workflows.configs.bsm_config    import BSM_WORKFLOW_CONFIG
from quantfin.workflows.configs.merton_config import MERTON_WORKFLOW_CONFIG
from quantfin.workflows.configs.heston_config import HESTON_WORKFLOW_CONFIG

AVAILABLE_MODELS = {
    "BSM": BSM_WORKFLOW_CONFIG,
    "Merton": MERTON_WORKFLOW_CONFIG,
    "Heston": HESTON_WORKFLOW_CONFIG,
}
PARAMS_DIR = "calibrated_params"
os.makedirs(PARAMS_DIR, exist_ok=True)

# Helper Functions for Parameter Management 
def get_params_filepath(ticker: str, snapshot_date: str, model_name: str) -> str:
    simple_model_name = model_name.split(" ")[0]
    return os.path.join(PARAMS_DIR, f"{ticker}_{simple_model_name}_{snapshot_date}.json")

def save_params(filepath: str, params: dict):
    with open(filepath, 'w') as f:
        json.dump(params, f, indent=4)

def load_params(filepath: str) -> dict | None:
    if not os.path.exists(filepath): return None
    with open(filepath, 'r') as f: return json.load(f)

#  Streamlit UI---
st.set_page_config(layout="wide", page_title="Option Calibration Dashboard")
st.title("Option Model Calibration & Analysis")

# Use session state to store results between interactions
if 'app_data' not in st.session_state:
    st.session_state.app_data = {}

with st.sidebar:
    st.header("Configuration")
    ticker = st.selectbox("Ticker", ('SPY', 'AAPL', 'META', 'GOOGL', 'TSLA'))
    available_dates = get_available_snapshot_dates(ticker)
    if not available_dates:
        st.error(f"No saved data for {ticker}."); st.stop()
    snapshot_date = st.selectbox("Snapshot Date", available_dates, index=len(available_dates)-1)
    model_selection = st.multiselect("Select Models", list(AVAILABLE_MODELS.keys()), default=["BSM", "Merton"])

run_button = st.sidebar.button("Run Analysis")

# Main App Logic 
if run_button:
    st.session_state.app_data = {} # Clear previous results on new run
    market_data = load_market_snapshot(ticker, snapshot_date)
    if market_data is None: st.error("Failed to load data."); st.stop()
    
    st.session_state.app_data['market_data'] = market_data
    st.session_state.app_data['calibrated_models'] = {}
    all_results = []

    for model_name_user in model_selection:
        model_config = AVAILABLE_MODELS[model_name_user]
        params_file = get_params_filepath(ticker, snapshot_date, model_name_user)
        
        # For the dashboard, we'll always run the calibration to show the process
        with st.spinner(f"Running workflow for {model_name_user}..."):
            workflow = DailyWorkflow(market_data=market_data, model_config=model_config)
            workflow.run()
            all_results.append(workflow.results)
            
            if workflow.results.get('Status') == 'Success':
                calibrated_params = workflow.results.get('Calibrated Params')
                save_params(params_file, calibrated_params)
                st.session_state.app_data['calibrated_models'][model_name_user] = model_config['model_class'](params=calibrated_params)
            else:
                st.error(f"Calibration failed for {model_name_user}. See console for details.")

    st.session_state.app_data['summary_df'] = pd.DataFrame(all_results)

# Display Area 
st.header(f"Analysis for {ticker} on {snapshot_date}")

if not st.session_state.app_data:
    st.info("Select parameters and click 'Run Analysis' in the sidebar to begin.")
else:
    st.subheader("Calibration Summary")
    st.dataframe(st.session_state.app_data['summary_df'])

    calibrated_models = st.session_state.app_data.get('calibrated_models', {})
    if not calibrated_models:
        st.warning("No models were successfully calibrated. Cannot display plots or pricing.")
    else:
        st.subheader("Volatility Smile Visualization")
        with st.spinner("Calculating IV surfaces and generating plot..."):
            market_data = st.session_state.app_data['market_data']
            
            stock = Stock(spot=market_data['spot_price'].iloc[0])
            calls = market_data[market_data['optionType'] == 'call']
            puts = market_data[market_data['optionType'] == 'put']
            implied_r, _ = fit_rate_and_dividend(calls, puts, stock.spot)
            rate = Rate(rate=implied_r)

            market_surface = VolatilitySurface(market_data).calculate_market_iv(stock, rate).surface
            
            model_surfaces = {}
            for name, model_instance in calibrated_models.items():
                technique = select_fastest_technique(model_instance)
                model_surfaces[name] = VolatilitySurface(market_data).calculate_model_iv(stock, rate, model_instance, technique).surface
            
            fig = plot_smiles_by_expiry(market_surface, model_surfaces, ticker, snapshot_date)
            st.pyplot(fig)

        #  Interactive Pricing and Greeks Section 
        st.header("On-Demand Pricing & Greeks")
        model_name_to_price = st.radio("Select Model for Pricing", list(calibrated_models.keys()))
        model_instance = calibrated_models[model_name_to_price]
        technique = select_fastest_technique(model_instance)

        col1, col2, col3 = st.columns(3)
        custom_strike = col1.number_input("Strike Price", value=stock.spot, step=1.0)
        custom_maturity = col2.number_input("Maturity (Years)", value=0.5, min_value=0.01, step=0.1)
        custom_type = col3.selectbox("Option Type", ("CALL", "PUT"))

        if st.button("Calculate Price & Greeks"):
            custom_option = Option(strike=custom_strike, maturity=custom_maturity, option_type=OptionType[custom_type])
            with st.spinner("Calculating..."):
                greeks = {
                    'Price': technique.price(custom_option, stock, model_instance, rate, **model_instance.params).price,
                    'Delta': technique.delta(custom_option, stock, model_instance, rate, **model_instance.params),
                    'Gamma': technique.gamma(custom_option, stock, model_instance, rate, **model_instance.params),
                    'Vega': technique.vega(custom_option, stock, model_instance, rate, **model_instance.params),
                    'Theta': technique.theta(custom_option, stock, model_instance, rate, **model_instance.params),
                    'Rho': technique.rho(custom_option, stock, model_instance, rate, **model_instance.params),
                }
                st.dataframe(pd.DataFrame([greeks]))