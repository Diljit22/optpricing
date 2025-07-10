from optpricing.models.heston import HestonModel

__doc__ = """
Calibration recipe for the Heston model.
"""

HESTON_WORKFLOW_CONFIG = {
    "name": "Heston",
    "model_class": HestonModel,
    "initial_guess": {
        "v0": 0.04,
        "kappa": 0.1,
        "theta": 0.04,
        "rho": -0.7,
        "vol_of_vol": 1,
    },
    "frozen": {},
    "bounds": {
        "v0": (0.001, 0.5),
        "kappa": (0.1, 10.0),
        "theta": (0.01, 0.5),
        "rho": (-0.99, 0.0),
        "vol_of_vol": (0.1, 1.5),
    },
}
