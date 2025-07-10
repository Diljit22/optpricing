import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from optpricing.dashboard.widgets import build_sidebar
from optpricing.data import get_live_option_chain, load_market_snapshot

__doc__ = """
An exploratory tool to view and analyze live or historical option chain data.
It provides visual summaries of volume, open interest, and implied volatility
distributions for a selected expiry date.
"""

st.set_page_config(layout="wide", page_title="Market Analytics")
st.title("Market Analytics")
st.caption("Explore live or historical option chain data with visual summaries.")

ticker, snapshot_date, _ = build_sidebar()


@st.cache_data(ttl=300)
def load_data(ticker_symbol, date_snapshot):
    """Cached function to load option chain data."""
    if date_snapshot == "Live Data":
        return get_live_option_chain(ticker_symbol)
    return load_market_snapshot(ticker_symbol, date_snapshot)


try:
    with st.spinner(f"Loading data for {ticker} on {snapshot_date}..."):
        chain_df = load_data(ticker, snapshot_date)
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

if chain_df is None or chain_df.empty:
    st.warning(f"No option chain data found for {ticker} on {snapshot_date}.")
    st.stop()

st.header("Option Chain Explorer")

# Filter by Expiry
expiries = sorted(chain_df["expiry"].unique())
expiry_str = [pd.to_datetime(e).strftime("%Y-%m-%d") for e in expiries]
selected_expiry_str = st.selectbox("Select Expiry Date", expiry_str)

if not selected_expiry_str:
    st.info("Please select an expiry date to view data.")
    st.stop()

# Filter dataframe by selected expiry
expiry_df = chain_df[chain_df["expiry"] == pd.to_datetime(selected_expiry_str)].copy()
calls = expiry_df[expiry_df["optionType"] == "call"].set_index("strike")
puts = expiry_df[expiry_df["optionType"] == "put"].set_index("strike")

tab1, tab2 = st.tabs(["📊 Summary Visuals", "📄 Raw Data"])

with tab1:
    st.subheader(f"Summary for Expiry: {selected_expiry_str}")
    c1, c2, c3 = st.columns(3)

    # Volume Bar Chart
    with c1:
        total_call_volume = calls["volume"].sum()
        total_put_volume = puts["volume"].sum()
        fig_vol = go.Figure(
            data=[
                go.Bar(name="Calls", x=["Volume"], y=[total_call_volume]),
                go.Bar(name="Puts", x=["Volume"], y=[total_put_volume]),
            ]
        )
        fig_vol.update_layout(
            barmode="group",
            title=(
                f"<b>Total Volume</b><br>"
                f"C: {total_call_volume:,.0f} | P: {total_put_volume:,.0f}"
            ),
            height=400,
        )
        st.plotly_chart(fig_vol, use_container_width=True)

    # IV Distribution Box Plot
    with c2:
        fig_iv = go.Figure()
        if not calls.empty:
            fig_iv.add_trace(
                go.Box(
                    y=calls["impliedVolatility"], name="Calls IV", marker_color="blue"
                )
            )
        if not puts.empty:
            fig_iv.add_trace(
                go.Box(y=puts["impliedVolatility"], name="Puts IV", marker_color="red")
            )
        fig_iv.update_layout(
            title="<b>Implied Volatility Distribution</b>",
            yaxis_tickformat=".2%",
            height=400,
        )
        st.plotly_chart(fig_iv, use_container_width=True)

    # Open Interest Pyramid
    with c3:
        common_strikes = sorted(calls.index.intersection(puts.index))
        call_oi = calls.loc[common_strikes]["openInterest"]
        put_oi = puts.loc[common_strikes]["openInterest"]

        fig_oi = go.Figure()
        # Call OI (positive side)
        fig_oi.add_trace(
            go.Bar(
                y=common_strikes,
                x=call_oi,
                name="Call OI",
                orientation="h",
                marker_color="blue",
            )
        )
        # Put OI (negative side)
        fig_oi.add_trace(
            go.Bar(
                y=common_strikes,
                x=-put_oi,
                name="Put OI",
                orientation="h",
                marker_color="red",
            )
        )
        fig_oi.update_layout(
            title="<b>Open Interest Pyramid</b>",
            barmode="relative",
            yaxis_title="Strike",
            xaxis=dict(tickformat=",.0f", title="Open Interest"),
            height=400,
            legend=dict(x=0.1, y=0.1),
        )
        st.plotly_chart(fig_oi, use_container_width=True)

with tab2:
    st.subheader(f"Raw Data for Expiry: {selected_expiry_str}")
    display_cols = [
        "strike",
        "optionType",
        "lastPrice",
        "bid",
        "ask",
        "volume",
        "openInterest",
        "impliedVolatility",
        "maturity",
    ]
    cols_to_show = [col for col in display_cols if col in expiry_df.columns]
    st.dataframe(expiry_df[cols_to_show])
