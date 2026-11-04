from optpricing.models.heston import HestonModel

__doc__ = """
Calibration recipe for the Heston model.

The initial guess satisfies the Feller condition (the previous one, kappa=0.1
with vol_of_vol=1, did not); the multi-start optimiser explores the rest of
the box.
"""

HESTON_WORKFLOW_CONFIG = {
    "name": "Heston",
    "model_class": HestonModel,
    "initial_guess": {
        "v0": 0.04,
        "kappa": 2.0,
        "theta": 0.04,
        "rho": -0.7,
        "vol_of_vol": 0.3,
    },
    "frozen": {},
    "bounds": {
        "v0": (0.001, 0.5),
        "kappa": (0.1, 10.0),
        "theta": (0.01, 0.5),
        "rho": (-0.99, 0.99),
        "vol_of_vol": (0.1, 1.5),
    },
}
