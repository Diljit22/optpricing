import streamlit as st

from quantfin.config import _config
from quantfin.dashboard.service import DashboardService
from quantfin.data import get_available_snapshot_dates
from quantfin.workflows.configs import ALL_MODEL_CONFIGS

st.set_page_config(layout="wide", page_title="QuantFin | Calibration")
st.title("Model Calibration & IV Surface Analysis")
st.caption(
    "Calibrate models to market data and visualize the resulting volatility smiles."
)
AVAILABLE_TICKERS = _config.get("default_tickers", ["SPY", "AAPL"])
ENABLED_MODELS = ["BSM", "Merton"]

all_model_names = list(ALL_MODEL_CONFIGS.keys())
disabled_models = [name for name in all_model_names if name not in ENABLED_MODELS]
AVAILABLE_MODELS_FOR_CALIBRATION = {
    name: config for name, config in ALL_MODEL_CONFIGS.items() if name in ENABLED_MODELS
}

with st.sidebar:
    st.header("Configuration")
    ticker = st.selectbox("Ticker", AVAILABLE_TICKERS)

    data_source_options = get_available_snapshot_dates(ticker)
    if data_source_options:
        data_source_options.insert(0, "Live Data")
    else:
        data_source_options = ["Live Data"]

    snapshot_date = st.selectbox("Snapshot Date", data_source_options)

    model_selection = st.multiselect(
        "Select Models to Calibrate",
        list(AVAILABLE_MODELS_FOR_CALIBRATION.keys()),
        default=["BSM"],
    )
    run_button = st.button("Run Calibration Analysis")

if run_button:
    selected_configs = {
        name: AVAILABLE_MODELS_FOR_CALIBRATION[name] for name in model_selection
    }
    with st.spinner("Initializing analysis..."):
        service = DashboardService(ticker, snapshot_date, selected_configs)
        try:
            service.run_calibrations()
            smile_fig, surface_fig = service.get_iv_plots()
            st.session_state.service = service
            st.session_state.smile_fig = smile_fig
            st.session_state.surface_fig = surface_fig
        except Exception as e:
            st.error(f"An error occurred during analysis: {e}")
            if "service" in st.session_state:
                del st.session_state.service

if "service" in st.session_state:
    service = st.session_state.service
    st.header(f"Analysis for {service.ticker} on {service.snapshot_date}")
    st.subheader("Calibration Summary")
    if service.summary_df is not None:
        st.dataframe(service.summary_df)
    else:
        st.error("Calibration failed for all selected models.")

    if "smile_fig" in st.session_state:
        st.subheader("Volatility Smile Visualization")
        st.pyplot(st.session_state.smile_fig)
        st.subheader("Implied Volatility Surface (3D)")
        st.plotly_chart(st.session_state.surface_fig, use_container_width=True)
else:
    st.info("Select parameters and click 'Run Analysis' in the sidebar to begin.")
