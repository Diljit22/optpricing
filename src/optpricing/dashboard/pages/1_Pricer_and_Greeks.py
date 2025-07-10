import pandas as pd
import streamlit as st

from optpricing.atoms import Option, OptionType, Rate, Stock
from optpricing.dashboard.widgets import build_sidebar
from optpricing.models import (
    BatesModel,
    BSMModel,
    CEVModel,
    CGMYModel,
    HestonModel,
    HyperbolicModel,
    KouModel,
    MertonJumpModel,
    NIGModel,
    SABRJumpModel,
    SABRModel,
    VarianceGammaModel,
)
from optpricing.techniques import (
    AmericanMonteCarloTechnique,
    ClosedFormTechnique,
    CRRTechnique,
    FFTTechnique,
    IntegrationTechnique,
    LeisenReimerTechnique,
    MonteCarloTechnique,
    PDETechnique,
    TOPMTechnique,
)

__doc__ = """
Provides an on-demand pricing tool where all model and market parameters can be
set manually. Ideal for sensitivity analysis and understanding model behavior.
"""

st.set_page_config(layout="wide", page_title="Pricer & Greeks")
st.title("On-Demand Pricer & Greek Analysis")
st.caption(
    "Price any option with any model and technique. "
    "Manually set all parameters to see their effect."
)
ticker, snapshot_date, model_name = build_sidebar()


MODEL_MAP = {
    "Bates": BatesModel,
    "BSM": BSMModel,
    "CEV": CEVModel,
    "CGMY": CGMYModel,
    "Heston": HestonModel,
    "Hyperbolic": HyperbolicModel,
    "Kou": KouModel,
    "Merton": MertonJumpModel,
    "NIG": NIGModel,
    "SABR": SABRModel,
    "SABRJump": SABRJumpModel,
    "VG": VarianceGammaModel,
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

model_class = MODEL_MAP[model_name]
try:
    default_params = getattr(model_class, "default_params", {})
    dummy_model_instance = model_class(params=default_params)
except Exception:
    dummy_model_instance = model_class.__new__(model_class)


st.subheader("Technique Selection")
col1, col2 = st.columns([0.5, 0.5])

with col1:
    # Dynamic Technique Selector based on model capabilities
    supported_techs = []
    if getattr(dummy_model_instance, "has_closed_form", False):
        supported_techs.append("Analytic/Closed-Form")
    if getattr(dummy_model_instance, "supports_cf", False):
        supported_techs.extend(["Integration", "FFT"])

    mc_supported = (
        getattr(dummy_model_instance, "supports_sde", False)
        or getattr(dummy_model_instance, "is_pure_levy", False)
        or getattr(dummy_model_instance, "has_exact_sampler", False)
    )

    if mc_supported and model_name != "Hyperbolic":
        supported_techs.append("Monte Carlo")

    if getattr(dummy_model_instance, "supports_pde", False):
        supported_techs.append("PDE")
    if model_name == "BSM":
        supported_techs.extend(["Leisen-Reimer", "CRR", "TOPM"])

    if not supported_techs:
        technique_name = None
        st.warning(f"No pricing techniques available for {model_name}.")
    else:
        technique_name = st.selectbox(
            "Select Technique",
            supported_techs,
            key="technique_name",
        )

with col2:
    exercise_style = "European"
    american_supported_techs = ["Leisen-Reimer", "CRR", "TOPM", "Monte Carlo"]
    if technique_name in american_supported_techs:
        exercise_style = st.radio(
            "Exercise Style",
            ["European", "American"],
            horizontal=True,
            key="exercise_style",
        )
    else:
        st.markdown("Exercise Style: **European**")


with st.expander("Market & Option Parameters", expanded=True):
    cols = st.columns(4)
    spot = cols[0].number_input("Spot Price", value=100.0, step=1.0)
    strike = cols[1].number_input("Strike Price", value=100.0, step=1.0)
    maturity = cols[2].number_input(
        "Maturity (Years)", value=1.0, min_value=0.01, step=0.1
    )
    option_type = cols[3].selectbox("Option Type", ("CALL", "PUT"))

    rate_val = cols[0].number_input(
        "Risk-Free Rate", value=0.05, step=0.01, format="%.3f"
    )
    div_val = cols[1].number_input(
        "Dividend Yield", value=0.02, step=0.01, format="%.3f"
    )


with st.expander(f"{model_name} Model Parameters", expanded=True):
    params = {}
    if hasattr(dummy_model_instance, "default_params"):
        param_defs = getattr(dummy_model_instance, "param_defs", {})
        num_cols = 4
        cols = st.columns(num_cols)

        params_to_show = {
            k: v
            for k, v in dummy_model_instance.default_params.items()
            if k not in ["max_sum_terms"]
        }

        for i, (p_name, p_default_value) in enumerate(params_to_show.items()):
            p_def = param_defs.get(p_name, {})
            params[p_name] = cols[i % num_cols].number_input(
                label=p_def.get("label", p_name.replace("_", " ").title()),
                value=float(p_default_value),
                min_value=p_def.get("min"),
                max_value=p_def.get("max"),
                step=p_def.get("step", 0.01),
                format="%.4f",
                key=f"{model_name}_{p_name}",
            )

if st.button("Calculate Price & Greeks", use_container_width=True):
    if not technique_name:
        st.error(f"Cannot perform calculation: No technique selected for {model_name}.")
    else:
        stock = Stock(spot=spot, dividend=div_val)
        rate = Rate(rate=rate_val)
        option = Option(
            strike=strike, maturity=maturity, option_type=OptionType[option_type]
        )

        # Merge UI params with non-UI default params
        full_params = model_class.default_params.copy()
        full_params.update(params)

        model = model_class(params=full_params)
        is_american_flag = exercise_style == "American"

        if technique_name == "Monte Carlo" and is_american_flag:
            technique = AmericanMonteCarloTechnique()
            st.info("Using Longstaff-Schwartz for American Monte Carlo.")
        else:
            try:
                technique = TECHNIQUE_MAP[technique_name](is_american=is_american_flag)
            except TypeError:
                technique = TECHNIQUE_MAP[technique_name]()

        # Prepare kwargs for techniques that need extra info (e.g., Heston's v0)
        pricing_kwargs = full_params.copy()

        with st.spinner("Calculating..."):
            try:
                results = {}
                results["Price"] = technique.price(
                    option, stock, model, rate, **pricing_kwargs
                ).price

                # Check for unstable Greeks
                skip_mc_greeks = model.has_jumps and isinstance(
                    technique, MonteCarloTechnique
                )
                if skip_mc_greeks:
                    st.info(
                        "Note: Greeks for jump-diffusion models under Monte Carlo "
                        "are unstable and not displayed."
                    )

                    for greek in ["Delta", "Gamma", "Vega", "Theta", "Rho"]:
                        results[greek] = "N/A"
                else:
                    results["Delta"] = technique.delta(
                        option, stock, model, rate, **pricing_kwargs
                    )
                    results["Gamma"] = technique.gamma(
                        option, stock, model, rate, **pricing_kwargs
                    )
                    results["Vega"] = technique.vega(
                        option, stock, model, rate, **pricing_kwargs
                    )
                    results["Theta"] = technique.theta(
                        option, stock, model, rate, **pricing_kwargs
                    )
                    results["Rho"] = technique.rho(
                        option, stock, model, rate, **pricing_kwargs
                    )

                st.subheader("Results")
                res_cols = st.columns([0.3, 0.7])
                with res_cols[0]:
                    st.metric(
                        label=f"{model_name} Price", value=f"${results['Price']:.5f}"
                    )

                with res_cols[1]:
                    greeks_df = pd.DataFrame(
                        {k: [v] for k, v in results.items() if k != "Price"}
                    )
                    st.dataframe(greeks_df)

            except Exception as e:
                st.error(f"Calculation failed: {e}")
                st.exception(e)
