from optpricing.models.nig import NIGModel

__doc__ = """
Calibration recipe for the NIG model.
"""

NIG_WORKFLOW_CONFIG = {
    "name": "NIG",
    "model_class": NIGModel,
    "initial_guess": {"alpha": 15.0, "beta": -5.0, "delta": 0.5},
    "frozen": {},
    "bounds": {
        "alpha": (1.0, 50.0),
        "beta": (-14.0, 14.0),  # Must be |beta| < alpha
        "delta": (0.1, 2.0),
    },
}
