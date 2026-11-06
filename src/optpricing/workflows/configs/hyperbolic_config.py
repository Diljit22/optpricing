from optpricing.models import HyperbolicModel

__doc__ = """
Calibration recipe for the Hyperbolic model.
"""

HYPERBOLIC_WORKFLOW_CONFIG = {
    "name": "Hyperbolic",
    "model_class": HyperbolicModel,
    "initial_guess": HyperbolicModel.default_params,
    # The risk-neutral drift correction cancels the location mu exactly, so
    # prices do not depend on it.
    "frozen": {"mu": 0.0},
    "bounds": {
        "alpha": (0.1, 50.0),
        "beta": (-49.0, 49.0),  # Must be less than alpha
        "delta": (0.01, 2.0),
    },
}
