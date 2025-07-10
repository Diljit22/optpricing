from optpricing.models.vg import VarianceGammaModel

__doc__ = """
Calibration recipe for the Variance Gamma model.
"""

VG_WORKFLOW_CONFIG = {
    "name": "Variance Gamma",
    "model_class": VarianceGammaModel,
    "initial_guess": {"sigma": 0.2, "nu": 0.1, "theta": -0.14},
    "frozen": {},
    "bounds": {"sigma": (0.01, 1.0), "nu": (0.001, 1.0), "theta": (-1.0, 1.0)},
}
