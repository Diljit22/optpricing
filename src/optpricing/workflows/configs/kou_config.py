from optpricing.models.kou import KouModel

__doc__ = """
Calibration recipe for the Kou double-exponential jump-diffusion model.

All five parameters are fitted. ``eta1`` must exceed 1 for the jump
compensator to exist, hence its lower bound.
"""

KOU_WORKFLOW_CONFIG = {
    "name": "Kou",
    "model_class": KouModel,
    "initial_guess": {
        "sigma": 0.15,
        "lambda": 1.0,
        "p_up": 0.4,
        "eta1": 10.0,
        "eta2": 5.0,
    },
    "frozen": {},
    "bounds": {
        "sigma": (0.01, 1.0),
        "lambda": (0.0, 10.0),
        "p_up": (0.0, 1.0),
        "eta1": (1.05, 50.0),
        "eta2": (0.1, 50.0),
    },
    "historical_prior": True,
}
