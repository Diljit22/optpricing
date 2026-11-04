from optpricing.models.bsm import BSMModel

__doc__ = """
Calibration recipe for the BSM model (a single flat volatility).
"""

BSM_WORKFLOW_CONFIG = {
    "name": "BSM",
    "model_class": BSMModel,
    "initial_guess": {"sigma": 0.2},
    "frozen": {},
    "bounds": {"sigma": (0.01, 2.0)},
}
