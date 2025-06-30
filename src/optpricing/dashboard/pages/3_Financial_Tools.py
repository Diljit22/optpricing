import pandas as pd
import streamlit as st

from optpricing.atoms import Option, OptionType, Rate, Stock, ZeroCouponBond
from optpricing.calibration import fit_jump_params_from_history, fit_rate_and_dividend
from optpricing.calibration.fit_market_params import find_atm_options
from optpricing.config import _config
from optpricing.data import get_live_option_chain, load_historical_returns
from optpricing.models import *
from optpricing.parity import ImpliedRateModel
from optpricing.techniques import *

st.set_page_config(layout="wide", page_title="optpricing | Live Tools")
st.title("Live Financial Tools & Pricer")

# --- Shared Sidebar for Ticker Selection ---
with st.sidebar:
    st.header("Configuration")
    AVAILABLE_TICKERS = _config.get("default_tickers", ["SPY", "AAPL"])
    ticker = st.selectbox("Select Ticker for Live Data", AVAILABLE_TICKERS)

# --- Live On-Demand Pricer ---
st.header("Live On-Demand Pricer")
st.caption("Price an option using live market data and your chosen model parameters.")

# Model and Technique Selection
pricer_cols = st.columns(2)
MODEL_MAP = {
    "BSM": BSMModel,
    "Merton": MertonJumpModel,
    "Heston": HestonModel,
    "Bates": BatesModel,
    "Kou": KouModel,
    "NIG": NIGModel,
    "VG": VarianceGammaModel,
    "CGMY": CGMYModel,
    "SABR": SABRModel,
    "CEV": CEVModel,
}
TECHNIQUE_MAP = {
    "Analytic/Closed-Form": ClosedFormTechnique,
    "Integration": IntegrationTechnique,
    "FFT": FFTTechnique,
    "Monte Carlo": MonteCarloTechnique,
    "PDE": PDETechnique,
    "Leisen-Reimer": LeisenReimerTechnique,
    "CRR": CRRTechnique,
    "TOPM": TOPMTechnique,
}

model_name_pricer = pricer_cols[0].selectbox(
    "Select Model",
    list(MODEL_MAP.keys()),
    key="pricer_model",
)

model_class = MODEL_MAP[model_name_pricer]
dummy_model_instance = (
    model_class() if hasattr(model_class, "default_params") else model_class
)
supported_techs = []
if dummy_model_instance.has_closed_form:
    supported_techs.append("Analytic/Closed-Form")
if dummy_model_instance.supports_cf:
    supported_techs.extend(["Integration", "FFT"])
if (
    dummy_model_instance.supports_sde
    or getattr(dummy_model_instance, "is_pure_levy", False)
    or getattr(dummy_model_instance, "has_exact_sampler", False)
):
    supported_techs.append("Monte Carlo")
technique_name = pricer_cols[1].selectbox(
    "Select Technique", supported_techs, key="pricer_tech"
)

# Dynamic Model Parameter Inputs
st.subheader(f"{model_name_pricer} Model Parameters")
params_pricer = {}
if hasattr(dummy_model_instance, "default_params"):
    param_defs = getattr(dummy_model_instance, "param_defs", {})
    num_cols = 4
    cols = st.columns(num_cols)
    for i, (p_name, p_default) in enumerate(
        dummy_model_instance.default_params.items()
    ):
        if p_name in ["max_sum_terms"]:
            continue
        p_def = param_defs.get(p_name, {})
        params_pricer[p_name] = cols[i % num_cols].number_input(
            label=p_def.get("label", p_name.replace("_", " ").title()),
            value=float(p_default),
            step=p_def.get("step", 0.01),
            key=f"pricer_{model_name_pricer}_{p_name}",
        )

# Option Definition
st.subheader("Option Parameters")
option_cols = st.columns(3)
strike_pricer = option_cols[0].number_input("Strike Price", value=100.0, step=1.0)
maturity_date_pricer = option_cols[1].date_input("Maturity Date")
option_type_pricer = option_cols[2].selectbox("Option Type", ("CALL", "PUT"))

if st.button("Calculate Live Price"):
    with st.spinner(f"Fetching live data for {ticker} and pricing..."):
        live_chain = get_live_option_chain(ticker)
        if live_chain is None or live_chain.empty:
            st.error(f"Could not fetch live data for {ticker}.")
        else:
            spot = live_chain["spot_price"].iloc[0]
            calls = live_chain[live_chain["optionType"] == "call"]
            puts = live_chain[live_chain["optionType"] == "put"]
            r, q = fit_rate_and_dividend(calls, puts, spot)

            stock = Stock(spot=spot, dividend=q)
            rate = Rate(rate=r)
            maturity_years = (
                pd.to_datetime(maturity_date_pricer) - pd.Timestamp.now()
            ).days / 365.25

            if maturity_years <= 0:
                st.error("Maturity date must be in the future.")
            else:
                option = Option(
                    strike=strike_pricer,
                    maturity=maturity_years,
                    option_type=OptionType[option_type_pricer],
                )
                full_params = model_class.default_params.copy()
                full_params.update(params_pricer)
                model = model_class(params=full_params)
                technique = TECHNIQUE_MAP[technique_name]()

                price_result = technique.price(
                    option, stock, model, rate, **full_params
                ).price
                result_msg = (
                    f"Calculated Option Price: ${price_result:.4f}\n"
                    f"(Spot: ${spot:.2f}, r: {r:.2%}, q: {q:.2%})"
                )
                st.success(f"Calculated Option Price: ${price_result:.4f}")
                st.success(f"(Spot: ${spot:.2f}, r: {r:.2%}, q: {q:.2%})")

st.divider()

# --- Put-Call Parity Tools ---
st.header("Put-Call Parity Tools")
if st.button("Fetch Live ATM Prices for Parity Check"):
    with st.spinner(f"Fetching live ATM data for {ticker}..."):
        live_chain = get_live_option_chain(ticker)
        if live_chain is not None and not live_chain.empty:
            spot = live_chain["spot_price"].iloc[0]
            calls = live_chain[live_chain["optionType"] == "call"]
            puts = live_chain[live_chain["optionType"] == "put"]
            atm_pairs = find_atm_options(calls, puts, spot)
            if not atm_pairs.empty:
                front_month_pair = atm_pairs.loc[atm_pairs["maturity"].idxmin()]
                st.session_state.call_p = front_month_pair["marketPrice_call"]
                st.session_state.put_p = front_month_pair["marketPrice_put"]
                st.session_state.spot_p = spot
                st.session_state.strike_p = front_month_pair["strike"]
                st.session_state.T_p = front_month_pair["maturity"]
            else:
                st.warning("No ATM pairs found for this ticker.")

c1, c2, c3, c4, c5 = st.columns(5)
call_p = c1.number_input(
    "Call Price", value=st.session_state.get("call_p", 10.0), key="call_p_input"
)
put_p = c2.number_input(
    "Put Price", value=st.session_state.get("put_p", 5.0), key="put_p_input"
)
spot_p = c3.number_input(
    "Spot Price ", value=st.session_state.get("spot_p", 100.0), key="spot_p_input"
)
strike_p = c4.number_input(
    "Strike Price ", value=st.session_state.get("strike_p", 100.0), key="strike_p_input"
)
T_p = c5.number_input(
    "Maturity ", value=st.session_state.get("T_p", 1.0), key="T_p_input", format="%.4f"
)

implied_rate_model = ImpliedRateModel(params={})
try:
    implied_r = implied_rate_model.price_closed_form(
        call_price=call_p, put_price=put_p, spot=spot_p, strike=strike_p, t=T_p, q=0
    )
    st.metric("Implied Risk-Free Rate (r)", f"{implied_r:.4%}")
except Exception as e:
    st.error(f"Could not calculate implied rate: {e}")

st.divider()

# Jump Parameter Fitter
st.header("Historical Jump Parameter Fitter")
TICKERS = ["SPY", "AAPL", "META", "GOOGL", "TSLA", "NVDA", "AMD", "MSFT", "AMZN", "JPM"]
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
    params["kappa"] = cols_vasicek[0].number_input("Mean Reversion (kappa)", value=0.86)
    params["theta"] = cols_vasicek[1].number_input("Long-Term Mean (theta)", value=0.09)
    params["sigma"] = cols_vasicek[2].number_input("Volatility (sigma)", value=0.02)
    model = VasicekModel(params=params)
else:  # CIR
    cols_cir = st.columns(3)
    params["kappa"] = cols_cir[0].number_input("Mean Reversion (kappa)", value=0.86)
    params["theta"] = cols_cir[1].number_input("Long-Term Mean (theta)", value=0.09)
    params["sigma"] = cols_cir[2].number_input("Volatility (sigma)", value=0.02)
    model = CIRModel(params=params)
if st.button("Price Zero-Coupon Bond"):
    bond = ZeroCouponBond(maturity=bond_maturity)
    r0_stock = Stock(spot=r0)
    dummy_rate = Rate(rate=0.0)
    technique = ClosedFormTechnique()
    price = technique.price(bond, r0_stock, model, dummy_rate).price
    st.metric(label=f"{model_name} ZCB Price", value=f"{price:.6f}")
