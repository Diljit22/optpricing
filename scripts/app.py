# scripts/app.py

import streamlit as st
import pandas as pd
from datetime import date

from quantfin.calibration.technique_selector import select_fastest_technique
from quantfin.data import get_available_snapshot_dates
from quantfin.dashboard.service import DashboardService
from quantfin.dashboard.widgets import render_parity_analysis_widget
from quantfin.atoms import Option, OptionType
from quantfin.workflows.configs import BSM_WORKFLOW_CONFIG, MERTON_WORKFLOW_CONFIG, HESTON_WORKFLOW_CONFIG

# --- App Configuration ---
st.set_page_config(layout="wide", page_title="QuantFin Dashboard")
st.title("Option Model Calibration & Analysis")

AVAILABLE_MODELS = {
    "BSM": BSM_WORKFLOW_CONFIG,
    "Merton": MERTON_WORKFLOW_CONFIG,
    "Heston": HESTON_WORKFLOW_CONFIG,
}

# --- Sidebar for User Input ---
with st.sidebar:
    st.header("Configuration")
    ticker = st.selectbox("Ticker", ('SPY', 'AAPL', 'META', 'GOOGL', 'TSLA'))
    
    data_source_options = ["Live Data"] + get_available_snapshot_dates(ticker)
    snapshot_date = st.selectbox("Data Source", data_source_options)
    
    model_selection = st.multiselect("Select Models to Calibrate", list(AVAILABLE_MODELS.keys()), default=["BSM", "Merton"])
    
    run_button = st.button("Run Analysis")

# --- Main Application Logic ---
if run_button:
    selected_configs = {name: AVAILABLE_MODELS[name] for name in model_selection}
    
    with st.spinner("Initializing analysis... This may take a moment for live data."):
        # Instantiate the service, which will handle all backend logic
        service = DashboardService(ticker, snapshot_date, selected_configs)
        
        # Run the core logic
        service.run_calibrations()
        
        # Generate plots
        smile_fig, surface_fig = service.get_iv_plots()
        
        # Store everything in the session state to persist it
        st.session_state.service = service
        st.session_state.smile_fig = smile_fig
        st.session_state.surface_fig = surface_fig

# --- Display Area ---
if 'service' not in st.session_state:
    st.info("Select parameters and click 'Run Analysis' in the sidebar to begin.")
else:
    # Retrieve the service and figures from the session state
    service = st.session_state.service
    
    st.header(f"Analysis for {service.ticker} on {service.snapshot_date}")
    
    # Display Calibration Summary
    st.subheader("Calibration Summary")
    if service.summary_df is not None:
        st.dataframe(service.summary_df)
    else:
        st.error("Calibration failed for all selected models.")

    # Display Plots
    if 'smile_fig' in st.session_state:
        st.subheader("Volatility Smile Visualization")
        st.pyplot(st.session_state.smile_fig)
        
        st.subheader("Implied Volatility Surface (3D)")
        st.plotly_chart(st.session_state.surface_fig, use_container_width=True)

    # Display Widgets
    if service.market_data is not None:
        render_parity_analysis_widget(service.market_data, service.stock, service.rate)

    # On-Demand Pricer
    st.header("On-Demand Pricer")
    if not service.calibrated_models:
        st.warning("No models were successfully calibrated. Cannot use on-demand pricer.")
    else:
        model_name = st.radio("Select Model for Pricing", list(service.calibrated_models.keys()))
        model_instance = service.calibrated_models[model_name]
        technique = select_fastest_technique(model_instance)
        col1, col2, col3 = st.columns(3)
        strike = col1.number_input("Strike", value=service.stock.spot, step=1.0)
        maturity = col2.number_input("Maturity (Years)", value=0.25, min_value=0.01, step=0.05)
        option_type = col3.selectbox("Type", ("CALL", "PUT"))
        
        if st.button("Price"):
            option = Option(strike=strike, maturity=maturity, option_type=OptionType[option_type])
            price = technique.price(option, service.stock, model_instance, service.rate, **model_instance.params).price
            st.metric(label=f"{model_name} Price", value=f"${price:.4f}")