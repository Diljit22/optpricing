from __future__ import annotations

import streamlit as st

from optpricing.config import _config
from optpricing.data import get_available_snapshot_dates
from optpricing.workflows.configs import ALL_MODEL_CONFIGS

__doc__ = """
Contains reusable Streamlit components (widgets) for displaying
specific analyses on the dashboard, such as put-call parity checks.
"""


def build_sidebar() -> tuple[str, str, str]:
    """
    Builds the consistent sidebar for the dashboard and returns the selections.

    Uses st.query_params to manage state for deep-linking.

    Returns
    -------
    tuple[str, str, str]
        A tuple containing the selected ticker, snapshot_date, and model_name.
    """
    with st.sidebar:
        st.header("Global Configuration")

        params = st.query_params
        default_ticker = params.get(
            "ticker", _config.get("default_tickers", ["SPY"])[0]
        )

        available_tickers = _config.get(
            "default_tickers",
            ["SPY", "AAPL", "TSLA", "NVDA"],
        )
        ticker_index = (
            available_tickers.index(default_ticker)
            if default_ticker in available_tickers
            else 0
        )
        ticker = st.selectbox(
            "Ticker",
            available_tickers,
            index=ticker_index,
            key="ticker_selector",
        )

        try:
            data_source_options = get_available_snapshot_dates(ticker)
        except FileNotFoundError:
            data_source_options = []

        data_source_options.insert(0, "Live Data")
        default_date = params.get("date", "Live Data")
        date_index = (
            data_source_options.index(default_date)
            if default_date in data_source_options
            else 0
        )
        snapshot_date = st.selectbox(
            "Snapshot Date",
            data_source_options,
            index=date_index,
            key="date_selector",
        )

        available_models = list(ALL_MODEL_CONFIGS.keys())
        default_model = params.get("model", "BSM")
        model_index = (
            available_models.index(default_model)
            if default_model in available_models
            else 0
        )
        model_name = st.selectbox(
            "Model",
            available_models,
            index=model_index,
            key="model_selector",
        )

        st.query_params["ticker"] = ticker
        st.query_params["date"] = snapshot_date
        st.query_params["model"] = model_name

        st.markdown("---")
        st.markdown("**Status:**")
        st.markdown(f"  **Ticker:** `{ticker}`")
        st.markdown(f"  **Date:** `{snapshot_date}`")
        st.markdown(f"  **Model:** `{model_name}`")

    return ticker, snapshot_date, model_name
