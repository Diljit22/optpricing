import streamlit as st

from optpricing.dashboard.plots import (
    plot_calendar_heatmap,
    plot_error_heatmap,
    plot_iv_surface_3d,
    plot_smiles_by_expiry,
)
from optpricing.dashboard.service import DashboardService
from optpricing.dashboard.widgets import build_sidebar
from optpricing.workflows.configs import ALL_MODEL_CONFIGS

__doc__ = """
Handles model calibration against historical data snapshots. This page allows
users to select a model and date, run the calibration, and visualize the
resulting implied volatility surfaces and calibration errors via heatmaps and
3D plots.
"""

st.set_page_config(layout="wide", page_title="Calibration & IV")
st.title("Model Calibration & IV Surface Analysis")
st.caption(
    "Calibrate models to market data and visualize the resulting volatility surfaces."
)

ticker, snapshot_date, _ = build_sidebar()

AVAILABLE_MODELS_FOR_CALIBRATION = {
    name: config
    for name, config in ALL_MODEL_CONFIGS.items()
    if name in ["BSM", "Merton"]
}

st.subheader("Calibration Controls")
model_selection = st.multiselect(
    "Select Models to Calibrate",
    list(AVAILABLE_MODELS_FOR_CALIBRATION.keys()),
    default=["BSM", "Merton"],
)

run_button = st.button("Run Calibration Analysis", use_container_width=True)

if run_button:
    if not model_selection:
        st.error("Please select at least one model to calibrate.")
    else:
        try:
            selected_configs = {
                name: AVAILABLE_MODELS_FOR_CALIBRATION[name] for name in model_selection
            }
            service = DashboardService(ticker, snapshot_date, selected_configs)
            service.run_calibrations()
            service.calculate_iv_surfaces()
            st.session_state.calibration_service = service
            st.toast("Calibration and analysis complete!", icon="✅")
        except Exception as e:
            st.error(f"An error occurred during analysis: {e}")
            if "calibration_service" in st.session_state:
                del st.session_state.calibration_service

if "calibration_service" in st.session_state:
    service: DashboardService = st.session_state.calibration_service
    st.header(f"Analysis for {service.ticker} on {service.snapshot_date}")
    st.subheader("Calibration Summary")
    if service.summary_df is not None:
        st.dataframe(service.summary_df)
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Volatility Smiles", "3D IV Surface", "Error Heatmap", "Calendar Heatmap"]
    )
    with tab1:
        st.plotly_chart(
            plot_smiles_by_expiry(service.market_surface, service.model_surfaces),
            use_container_width=True,
        )
    with tab2:
        st.plotly_chart(
            plot_iv_surface_3d(service.market_surface, service.model_surfaces),
            use_container_width=True,
        )
    with tab3:
        if not service.calibrated_models:
            st.info("No models were successfully calibrated.")
        else:
            model_to_plot = st.selectbox(
                "Select Model for Error Heatmap", list(service.calibrated_models.keys())
            )
            if model_to_plot and model_to_plot in service.model_surfaces:
                st.plotly_chart(
                    plot_error_heatmap(
                        service.market_surface,
                        service.model_surfaces[model_to_plot],
                        model_to_plot,
                    ),
                    use_container_width=True,
                )
    with tab4:
        st.plotly_chart(
            plot_calendar_heatmap(service.market_data), use_container_width=True
        )
