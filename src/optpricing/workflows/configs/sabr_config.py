from optpricing.models import SABRModel

__doc__ = """
Calibration recipe for the SABR model.

This configuration fits all SABR parameters to the market option prices.
"""

SABR_WORKFLOW_CONFIG = {
    "name": "SABR",
    "model_class": SABRModel,
    "initial_guess": SABRModel.default_params,
    "frozen": {},
    "bounds": {
        "alpha": (0.01, 2.0),
        "beta": (0.0, 1.0),
        "rho": (-0.99, 0.99),
    },
}
