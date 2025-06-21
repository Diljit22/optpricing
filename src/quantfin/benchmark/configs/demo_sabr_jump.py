
from quantfin.atoms.option import Option, OptionType
from quantfin.atoms.rate import Rate
from quantfin.atoms.stock import Stock
from quantfin.benchmark.configs.base_config import BenchmarkConfig
from quantfin.models.sabr_jump import SABRJumpModel
from quantfin.techniques.monte_carlo import MonteCarloTechnique

v0 = 0.20
sabr_jump_params = {
    "alpha": 0.5, "beta": 0.8, "rho": -0.6,
    "lambda": 0.4, "mu_j": -0.1, "sigma_j": 0.15
}
model = SABRJumpModel(params=sabr_jump_params)

stock = Stock(spot=100.0, dividend=0.01)
rate = Rate(rate=0.03)
options_to_price = [
    Option(strike=100.0, maturity=1.0, option_type=OptionType.CALL),
    Option(strike=100.0, maturity=1.0, option_type=OptionType.PUT),
]

config = BenchmarkConfig(
    name="SABR Jump Model",
    model=model,
    model_params=sabr_jump_params,
    techniques=[
        (MonteCarloTechnique(n_paths=50000, n_steps=100, seed=42), "MC")
    ],
    stock=stock,
    rate=rate,
    options=options_to_price,
    # Pass the initial volatility v0 to the MC technique
    technique_kwargs={"v0": v0}
)
