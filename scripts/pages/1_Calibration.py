import streamlit as st

from quantfin.dashboard.service import DashboardService
from quantfin.data import get_available_snapshot_dates
from quantfin.workflows.configs import (
    BSM_WORKFLOW_CONFIG,
    HESTON_WORKFLOW_CONFIG,
    MERTON_WORKFLOW_CONFIG,
)

st.set_page_config(layout="wide", page_title="QuantFin | Calibration")
st.title("Model Calibration & IV Surface Analysis")
st.caption("Calibrate models to market data and visualize the resulting volatility smiles.")

AVAILABLE_MODELS = {"BSM": BSM_WORKFLOW_CONFIG, "Merton": MERTON_WORKFLOW_CONFIG, "Heston": HESTON_WORKFLOW_CONFIG}

with st.sidebar:
    st.header("Configuration")
    ticker = st.selectbox("Ticker", ('SPY', 'AAPL', 'META', 'GOOGL', 'TSLA'))
    data_source_options = get_available_snapshot_dates(ticker)
    snapshot_date = st.selectbox("Snapshot Date", data_source_options)
    model_selection = st.multiselect("Select Models to Calibrate", list(AVAILABLE_MODELS.keys()), default=["BSM", "Merton"])
    run_button = st.button("Run Calibration Analysis")

if run_button:
    selected_configs = {name: AVAILABLE_MODELS[name] for name in model_selection}
    with st.spinner("Initializing analysis..."):
        service = DashboardService(ticker, snapshot_date, selected_configs)
        service.run_calibrations()
        smile_fig, surface_fig = service.get_iv_plots()
        st.session_state.service = service
        st.session_state.smile_fig = smile_fig
        st.session_state.surface_fig = surface_fig

if 'service' in st.session_state:
    service = st.session_state.service
    st.header(f"Analysis for {service.ticker} on {service.snapshot_date}")
    st.subheader("Calibration Summary")
    if service.summary_df is not None:
        st.dataframe(service.summary_df)
    else:
        st.error("Calibration failed for all selected models.")

    if 'smile_fig' in st.session_state:
        st.subheader("Volatility Smile Visualization")
        st.pyplot(st.session_state.smile_fig)
        st.subheader("Implied Volatility Surface (3D)")
        st.plotly_chart(st.session_state.surface_fig, use_container_width=True)
else:
    st.info("Select parameters and click 'Run Analysis' in the sidebar to begin.")
