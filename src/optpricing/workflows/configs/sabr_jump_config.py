from optpricing.models import SABRJumpModel

__doc__ = """
Calibration recipe for the SABR Jump model.

The model has neither a closed form nor a characteristic function, so it is
priced by Monte Carlo and only calibrated when Monte Carlo pricing is
explicitly allowed (``--allow-mc``).
"""

SABR_JUMP_WORKFLOW_CONFIG = {
    "name": "SABRJump",
    "model_class": SABRJumpModel,
    "bounds": {
        "alpha": (0.01, 2.0),
        "beta": (0.0, 1.0),
        "rho": (-0.99, 0.99),
        "nu": (0.01, 3.0),
        "lambda": (0.0, 5.0),
        "mu_j": (-1.0, 1.0),
        "sigma_j": (0.01, 1.0),
    },
    "initial_guess": SABRJumpModel.default_params,
    "frozen": {},
    "requires_mc": True,
}
