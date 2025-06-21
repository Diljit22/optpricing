import numpy as np
import pandas as pd
import streamlit as st

from quantfin.atoms import Rate, Stock
from quantfin.parity import ParityModel


def render_parity_analysis_widget(market_data: pd.DataFrame, stock: Stock, rate: Rate):
    """Creates a Streamlit container to display put-call parity analysis."""
    st.subheader("Put-Call Parity Analysis")

    parity_model = ParityModel()

    expiries = sorted(market_data['expiry'].unique())
    selected_expiry = st.select_slider("Select Expiry to Analyze", options=[pd.to_datetime(e).strftime('%Y-%m-%d') for e in expiries])

    expiry_df = market_data[market_data['expiry'] == pd.to_datetime(selected_expiry)].copy()

    calls = expiry_df[expiry_df['optionType'] == 'call'].set_index('strike')
    puts = expiry_df[expiry_df['optionType'] == 'put'].set_index('strike')

    # Find common strikes
    common_strikes = calls.index.intersection(puts.index)
    if len(common_strikes) == 0:
        st.warning("No common strikes found for this expiry to check parity.")
        return

    calls = calls.loc[common_strikes]
    puts = puts.loc[common_strikes]

    # C - P
    price_diff = calls['last_price'] - puts['last_price']

    # S*exp(-qT) - K*exp(-rT)
    T = calls['maturity'].iloc[0]
    r = rate.get_rate(T)
    q = stock.dividend

    parity_diff = stock.spot * np.exp(-q * T) - calls.index.values * np.exp(-r * T)

    error = price_diff - parity_diff

    results_df = pd.DataFrame({
        'C - P': price_diff,
        'S*e⁻qT - K*e⁻rT': parity_diff,
        'Arbitrage Gap ($)': error
    })

    st.dataframe(results_df)
    st.caption("A non-zero 'Arbitrage Gap' suggests a violation of put-call parity (or stale data).")
