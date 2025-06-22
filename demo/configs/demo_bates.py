from quantfin.atoms.option import Option, OptionType
from quantfin.atoms.rate import Rate
from quantfin.atoms.stock import Stock
from quantfin.benchmark.configs.base_config import BenchmarkConfig
from quantfin.models.bates import BatesModel
from quantfin.techniques.fft import FFTTechnique
from quantfin.techniques.integration import IntegrationTechnique
from quantfin.techniques.monte_carlo import MonteCarloTechnique

v0, kappa, theta, rho, vol_of_vol = 0.04, 2.0, 0.04, -0.7, 0.5
lambda_, mu_j, sigma_j = 0.5, -0.1, 0.15
bates_params = {
    "v0": v0,
    "kappa": kappa,
    "theta": theta,
    "rho": rho,
    "vol_of_vol": vol_of_vol,
    "lambda": lambda_,
    "mu_j": mu_j,
    "sigma_j": sigma_j,
}
model = BatesModel(params=bates_params)

stock = Stock(spot=100.0, dividend=0.01)
rate = Rate(rate=0.03)
options_to_price = [
    Option(strike=100.0, maturity=1.0, option_type=OptionType.CALL),
    Option(strike=100.0, maturity=1.0, option_type=OptionType.PUT),
]

config = BenchmarkConfig(
    name="Bates Stochastic Volatility Jump Model",
    model=model,
    model_params=bates_params,
    techniques=[
        (IntegrationTechnique(), "Integration"),
        (FFTTechnique(n=14), "FFT"),
        (MonteCarloTechnique(n_paths=50000, n_steps=500, seed=42), "MC"),
    ],
    stock=stock,
    rate=rate,
    options=options_to_price,
    technique_kwargs={"v0": v0},
)
