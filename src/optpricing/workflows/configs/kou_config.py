from optpricing.models.kou import KouModel

__doc__ = """
Calibration recipe for the Kou model.
"""

KOU_WORKFLOW_CONFIG = {
    "name": "Kou",
    "model_class": KouModel,
    "historical_params": ["sigma", "lambda"],
    "initial_guess": {
        "sigma": 0.15,
        "lambda": 1.0,
        "p_up": 0.6,
        "eta1": 10.0,
        "eta2": 5.0,
    },
    "frozen": ["lambda", "p_up", "eta1", "eta2"],
    "bounds": {"sigma": (0.01, 1.0)},
}
