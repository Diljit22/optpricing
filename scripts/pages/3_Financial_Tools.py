
import pandas as pd
import streamlit as st

from quantfin.atoms import Rate, Stock, ZeroCouponBond
from quantfin.calibration import fit_jump_params_from_history
from quantfin.data import load_historical_returns
from quantfin.models import CIRModel, VasicekModel
from quantfin.parity import ImpliedRateModel
from quantfin.techniques import ClosedFormTechnique

st.set_page_config(layout="wide", page_title="QuantFin | Tools")
st.title("Financial Utilities & Tools")

# Jump Parameter Fitter
st.header("Historical Jump Parameter Fitter")
TICKERS = ['SPY', 'AAPL', 'META', 'GOOGL', 'TSLA', 'NVDA', 'AMD', 'MSFT', 'AMZN', 'JPM']
ticker_jump = st.selectbox("Select Ticker for Jump Analysis", TICKERS)
if st.button("Fit Jump Parameters"):
    with st.spinner(f"Loading 10y returns for {ticker_jump} and fitting..."):
        try:
            returns = load_historical_returns(ticker_jump, period="10y")
            jump_params = fit_jump_params_from_history(returns)
            st.dataframe(pd.DataFrame([jump_params]))
        except Exception as e:
            st.error(f"Could not fit parameters: {e}")

# Rate Model Pricer
st.header("Term Structure Model Pricer")
col1, col2, col3 = st.columns(3)
model_name = col1.selectbox("Select Rate Model", ["Vasicek", "CIR"])
r0 = col2.number_input("Initial Short Rate (r0)", value=0.05, step=0.01)
bond_maturity = col3.number_input("Bond Maturity (T)", value=1.0, step=0.5)
params = {}
if model_name == "Vasicek":
    cols_vasicek = st.columns(3)
    params['kappa'] = cols_vasicek[0].number_input("Mean Reversion (kappa)", value=0.86)
    params['theta'] = cols_vasicek[1].number_input("Long-Term Mean (theta)", value=0.09)
    params['sigma'] = cols_vasicek[2].number_input("Volatility (sigma)", value=0.02)
    model = VasicekModel(params=params)
else: # CIR
    cols_cir = st.columns(3)
    params['kappa'] = cols_cir[0].number_input("Mean Reversion (kappa)", value=0.86)
    params['theta'] = cols_cir[1].number_input("Long-Term Mean (theta)", value=0.09)
    params['sigma'] = cols_cir[2].number_input("Volatility (sigma)", value=0.02)
    model = CIRModel(params=params)
if st.button("Price Zero-Coupon Bond"):
    bond = ZeroCouponBond(maturity=bond_maturity)
    r0_stock = Stock(spot=r0)
    dummy_rate = Rate(rate=0.0)
    technique = ClosedFormTechnique()
    price = technique.price(bond, r0_stock, model, dummy_rate).price
    st.metric(label=f"{model_name} ZCB Price", value=f"{price:.6f}")

st.header("Put-Call Parity Tools")
c1, c2, c3, c4, c5 = st.columns(5)
call_p = c1.number_input("Call Price", value=10.0)
put_p = c2.number_input("Put Price", value=5.0)
spot_p = c3.number_input("Spot Price ", value=100.0)
strike_p = c4.number_input("Strike Price ", value=100.0)
T_p = c5.number_input("Maturity ", value=1.0)

implied_rate_model = ImpliedRateModel(params={'eps': 1e-9, 'max_iter': 100})
try:
    implied_r = implied_rate_model.price_closed_form(call_price=call_p, put_price=put_p, spot=spot_p, strike=strike_p, t=T_p, q=0)
    st.metric("Implied Risk-Free Rate (r)", f"{implied_r:.4%}")
except Exception as e:
    st.error(f"Could not calculate implied rate: {e}")
