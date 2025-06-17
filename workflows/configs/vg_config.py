from models.vg import VarianceGammaModel

VG_WORKFLOW_CONFIG = {
    "model_name": "Variance Gamma",
    "model_class": VarianceGammaModel,
    "initial_guess": {"sigma": 0.2, "nu": 0.1, "theta": -0.14},
    "frozen": {}, # No frozen params, fit all to the smile
    "bounds": {
        "sigma": (0.01, 1.0),
        "nu": (0.001, 1.0),
        "theta": (-1.0, 1.0)
    }
}