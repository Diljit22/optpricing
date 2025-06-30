from optpricing.models import BatesModel

__doc__ = """
Calibration recipe for the Bates (Heston + Jumps) model.

This configuration fits all stochastic volatility and jump parameters
simultaneously to the market option prices.
"""

BATES_WORKFLOW_CONFIG = {
    "name": "Bates",
    "model_class": BatesModel,
    "initial_guess": BatesModel.default_params,
    "frozen": {},
    "bounds": {
        "v0": (0.001, 0.5),
        "kappa": (0.1, 10.0),
        "theta": (0.01, 0.5),
        "rho": (-0.99, 0.99),
        "vol_of_vol": (0.1, 1.5),
        "lambda": (0.0, 5.0),
        "mu_j": (-1.0, 1.0),
        "sigma_j": (0.01, 1.0),
    },
}
