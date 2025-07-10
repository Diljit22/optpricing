import pandas as pd
import plotly.graph_objects as go
import scipy.stats as stats
import streamlit as st

from optpricing.calibration import fit_jump_params_from_history
from optpricing.dashboard.widgets import build_sidebar
from optpricing.data import load_historical_returns

__doc__ = """
A quantitative research page for analyzing historical return distributions.
It includes a QQ-Plot to assess normality and a tool to fit jump-diffusion
parameters from historical data using the method of moments.
"""

st.set_page_config(layout="wide", page_title="Quant Research")
st.title("Quantitative Research")
st.caption(
    "Tools for analyzing historical return distributions and fitting model parameters."
)

ticker, _, _ = build_sidebar()

st.header("Distributional Analysis (QQ-Plot)")
st.markdown(
    "A Quantile-Quantile (QQ) plot helps us visually assess if historical returns "
    "follow a theoretical distribution (e.g., Normal for BSM)."
)
if st.button(f"Generate QQ-Plot for {ticker}"):
    with st.spinner(f"Loading 10y returns for {ticker}..."):
        try:
            returns = load_historical_returns(ticker, period="10y")
            if returns is None or returns.empty:
                st.error(f"No historical data found for {ticker}.")
            else:
                (osm, osr), (slope, intercept, r) = stats.probplot(returns, dist="norm")
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(x=osm, y=osr, mode="markers", name="Data Quantiles")
                )
                fig.add_trace(
                    go.Scatter(
                        x=osm,
                        y=slope * osm + intercept,
                        mode="lines",
                        name="Normal Line",
                    )
                )
                fig.update_layout(
                    title=f"<b>QQ-Plot of {ticker} Returns vs. Normal Distribution</b>",
                    xaxis_title="Theoretical Quantiles",
                    yaxis_title="Sample Quantiles",
                    annotations=[
                        dict(
                            x=0.05,
                            y=0.95,
                            showarrow=False,
                            text=f"R-squared = {r**2:.4f}",
                            xref="paper",
                            yref="paper",
                        )
                    ],
                )
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Could not generate QQ-Plot: {e}")

st.divider()

st.header("Historical Jump Parameter Fitter")
st.markdown(
    "This tool uses the method of moments on historical log-returns to estimate the "
    "parameters (λ, μ_j, σ_j) for a Merton-style jump-diffusion process."
)
if st.button(f"Fit Jump Parameters for {ticker}"):
    with st.spinner(f"Loading 10y returns for {ticker}..."):
        try:
            returns = load_historical_returns(ticker, period="10y")
            if returns is None or returns.empty:
                st.error(f"No historical data found for {ticker}.")
            else:
                jump_params = fit_jump_params_from_history(returns)
                st.dataframe(pd.DataFrame([jump_params]))
        except Exception as e:
            st.error(f"Could not fit parameters: {e}")
