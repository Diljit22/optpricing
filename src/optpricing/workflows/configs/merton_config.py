from optpricing.models.merton_jump import MertonJumpModel

__doc__ = """
Calibration recipe for the Merton jump-diffusion model.

All four parameters are fitted to the option chain. With ``historical_prior``
the jump intensity and jump-size moments estimated from the ticker's
historical returns seed the optimiser (they used to be frozen, which left
nothing to fit).
"""

MERTON_WORKFLOW_CONFIG = {
    "name": "Merton",
    "model_class": MertonJumpModel,
    "initial_guess": {
        "sigma": 0.15,
        "lambda": 0.5,
        "mu_j": -0.1,
        "sigma_j": 0.15,
    },
    "frozen": {},
    "bounds": {
        "sigma": (0.01, 1.0),
        "lambda": (0.0, 5.0),
        "mu_j": (-1.0, 1.0),
        "sigma_j": (0.01, 1.0),
    },
    "historical_prior": True,
}
