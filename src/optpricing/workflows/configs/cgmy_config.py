from optpricing.models import CGMYModel

__doc__ = """
Calibration recipe for the CGMY model.
"""

CGMY_WORKFLOW_CONFIG = {
    "name": "CGMY",
    "model_class": CGMYModel,
    "initial_guess": CGMYModel.default_params,
    "frozen": {},
    "bounds": {
        "C": (0.001, 1.0),
        "G": (0.1, 20.0),
        "M": (0.1, 20.0),
        "Y": (0.0, 1.99),
    },
}
