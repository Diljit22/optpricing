from optpricing.models import SABRJumpModel

__doc__ = """
Calibration recipe for the SABR Jump model.

This configuration fits all SABR and jump parameters simultaneously
to the market option prices.
"""

SABR_JUMP_WORKFLOW_CONFIG = {
    "name": "SABRJump",
    "model_class": SABRJumpModel,
    "bounds": {
        "alpha": (0.01, 2.0),
        "beta": (0.0, 1.0),
        "rho": (-0.99, 0.99),
        "lambda": (0.0, 5.0),
        "mu_j": (-1.0, 1.0),
        "sigma_j": (0.01, 1.0),
    },
    "initial_guess": SABRJumpModel.default_params,
    "frozen": {"v0": SABRJumpModel.default_params.get("v0", 0.5)},
}
