import streamlit as st

from optpricing.calibration import CalibrationSettings
from optpricing.dashboard.plots import (
    plot_calendar_heatmap,
    plot_error_heatmap,
    plot_iv_surface_3d,
    plot_smiles_by_expiry,
)
from optpricing.dashboard.service import DashboardService
from optpricing.dashboard.widgets import build_sidebar
from optpricing.workflows import load_tuning
from optpricing.workflows.configs import ALL_MODEL_CONFIGS, calibratable_model_names

__doc__ = """
Handles model calibration against historical data snapshots. This page allows
users to select models and a date, run the calibration, and visualize the
resulting implied volatility surfaces and calibration errors via heatmaps and
3D plots.
"""

st.set_page_config(layout="wide", page_title="Calibration & IV")
st.title("Model Calibration & IV Surface Analysis")
st.caption(
    "Calibrate models to market data and visualize the resulting volatility surfaces."
)

ticker, snapshot_date, _ = build_sidebar()

st.subheader("Calibration Controls")
allow_mc = st.checkbox(
    "Allow Monte Carlo pricing (adds simulation-only models; slow)", value=False
)
AVAILABLE_MODELS_FOR_CALIBRATION = {
    name: ALL_MODEL_CONFIGS[name]
    for name in calibratable_model_names(allow_mc=allow_mc)
}
model_selection = st.multiselect(
    "Select Models to Calibrate",
    list(AVAILABLE_MODELS_FOR_CALIBRATION.keys()),
    default=["BSM", "Merton", "Heston"],
)

tuning = load_tuning(ticker)
c1, c2, c3 = st.columns(3)
with c1:
    use_tuned = st.checkbox(
        "Use tuned settings",
        value=tuning is not None,
        disabled=tuning is None,
        help="Settings cached by `optpricing tune --ticker ...`"
        if tuning is None
        else f"Tuned on {len(tuning.dates)} snapshots ({tuning.created[:10]})",
    )
with c2:
    objective = st.selectbox("Objective", ["iv", "price"], index=0)
with c3:
    n_starts = st.slider("Starting points", min_value=1, max_value=8, value=2)

if tuning is not None:
    with st.expander(f"Tuning leaderboard for {ticker}"):
        st.dataframe(tuning.to_frame())

run_button = st.button("Run Calibration Analysis", use_container_width=True)

if run_button:
    if not model_selection:
        st.error("Please select at least one model to calibrate.")
    else:
        try:
            selected_configs = {
                name: AVAILABLE_MODELS_FOR_CALIBRATION[name] for name in model_selection
            }
            service = DashboardService(
                ticker,
                snapshot_date,
                selected_configs,
                settings=CalibrationSettings(objective=objective, n_starts=n_starts),
                use_tuned=use_tuned,
                allow_mc=allow_mc,
            )
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
    if service.final_stock is not None and service.final_rate is not None:
        stock, rate = service.final_stock, service.final_rate
        st.caption(
            f"Spot {stock.spot:.2f} | dividend yield {stock.dividend:.2%} | "
            f"rate {rate.get_rate():.2%}"
        )
    st.subheader("Calibration Leaderboard")
    if service.summary_df is not None:
        st.dataframe(service.summary_df)
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "Volatility Smiles",
            "3D IV Surface",
            "Error Heatmap",
            "Error by Bucket",
            "Calendar Heatmap",
        ]
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
        if not service.bucket_tables:
            st.info("No bucket metrics available.")
        else:
            model_for_buckets = st.selectbox(
                "Select Model for Bucket Metrics", list(service.bucket_tables.keys())
            )
            if model_for_buckets:
                st.dataframe(service.bucket_tables[model_for_buckets])
    with tab5:
        st.plotly_chart(
            plot_calendar_heatmap(service.market_data), use_container_width=True
        )
