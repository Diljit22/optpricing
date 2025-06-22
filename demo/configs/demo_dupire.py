import numpy as np

from quantfin.atoms.option import Option, OptionType
from quantfin.atoms.rate import Rate
from quantfin.atoms.stock import Stock
from quantfin.benchmark.configs.base_config import BenchmarkConfig
from quantfin.models.dupire_local import DupireLocalVolModel
from quantfin.techniques.monte_carlo import MonteCarloTechnique


# Define a mock volatility surface ---
# In a real application, this would be a 2D interpolator object
# calibrated to market data. For this demo, we create a simple function.
# This surface has a "smile": vol is higher for low/high strikes.
def mock_vol_surface(t: float, s: np.ndarray) -> np.ndarray:
    """A simple smile function: vol = 0.2 - 0.1*(S-100)/100 + 0.2*((S-100)/100)^2"""

    # Add a small epsilon to prevent log(0) errors.
    # This ensures that even if a path hits exactly zero, we get a
    # valid (though large) log-moneyness instead of -inf.
    epsilon = 1e-12
    safe_s = np.maximum(s, epsilon)

    log_moneyness = np.log(safe_s / 100.0)
    return 0.20 + 0.1 * log_moneyness + 0.5 * log_moneyness**2


# Create the Dupire Model ---
# The 'params' dictionary holds the callable surface object.
dupire_params = {"vol_surface": mock_vol_surface}
model = DupireLocalVolModel(params=dupire_params)

# Market and options
stock = Stock(spot=100.0, dividend=0.01)
rate = Rate(rate=0.03)
options_to_price = [
    Option(strike=90.0, maturity=1.0, option_type=OptionType.CALL),
    Option(strike=100.0, maturity=1.0, option_type=OptionType.CALL),
    Option(strike=110.0, maturity=1.0, option_type=OptionType.CALL),
]

# Create the config object
config = BenchmarkConfig(
    name="Dupire Local Volatility Model",
    model=model,
    model_params={"surface": "smile function"},  # For display
    techniques=[(MonteCarloTechnique(n_paths=50000, n_steps=252, seed=42), "MC")],
    stock=stock,
    rate=rate,
    options=options_to_price,
)
