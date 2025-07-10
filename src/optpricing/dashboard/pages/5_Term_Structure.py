import numpy as np
import plotly.graph_objects as go
import streamlit as st

from optpricing.atoms import Rate, Stock, ZeroCouponBond
from optpricing.dashboard.widgets import build_sidebar
from optpricing.models import CIRModel, VasicekModel
from optpricing.techniques import ClosedFormTechnique

__doc__ = """
A page for pricing and visualizing interest rate term structures. It allows
users to price Zero-Coupon Bonds and generate yield curves using short-rate
models like Vasicek and Cox-Ingersoll-Ross (CIR).
"""

st.set_page_config(layout="wide", page_title="Term Structure")
st.title("Term Structure Models")
st.caption(
    "Price Zero-Coupon Bonds and visualize yield curves using short-rate models."
)

build_sidebar()

st.header("Short-Rate Model Pricer")
col1, col2, col3, col4 = st.columns(4)
model_name = col1.selectbox("Select Rate Model", ["Vasicek", "CIR"])
r0 = col2.number_input("Initial Short Rate (r0)", value=0.03, step=0.005, format="%.3f")

params = {}
if model_name == "Vasicek":
    params["kappa"] = col2.number_input("Mean Reversion (κ)", value=0.86)
    params["theta"] = col3.number_input("Long-Term Mean (θ)", value=0.05)
    params["sigma"] = col4.number_input("Volatility (σ)", value=0.02)
    model = VasicekModel(params=params)
else:  # CIR
    params["kappa"] = col2.number_input("Mean Reversion (κ)", value=0.5)
    params["theta"] = col3.number_input("Long-Term Mean (θ)", value=0.04)
    params["sigma"] = col4.number_input("Volatility (σ)", value=0.1)
    model = CIRModel(params=params)

st.subheader("Generated Yield Curve")
maturities = np.linspace(0.1, 30, 100)
prices = []
technique = ClosedFormTechnique()
r0_stock = Stock(spot=r0)
dummy_rate = Rate(rate=0.0)

for T in maturities:
    bond = ZeroCouponBond(maturity=T)
    price = technique.price(bond, r0_stock, model, dummy_rate).price
    prices.append(price)

prices = np.array(prices)
yields = -np.log(prices) / maturities

fig = go.Figure(data=go.Scatter(x=maturities, y=yields, mode="lines"))
fig.update_layout(
    title=f"<b>{model_name} Yield Curve</b>",
    xaxis_title="Maturity (Years)",
    yaxis_title="Yield",
    yaxis_tickformat=".2%",
)
st.plotly_chart(fig, use_container_width=True)
