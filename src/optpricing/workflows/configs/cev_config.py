from optpricing.models import CEVModel

__doc__ = """
Calibration recipe for the Constant Elasticity of Variance (CEV) model.
"""

CEV_WORKFLOW_CONFIG = {
    "name": "CEV",
    "model_class": CEVModel,
    "initial_guess": CEVModel.default_params,
    "frozen": {},
    "bounds": {
        "sigma": (0.01, 2.0),
        "gamma": (0.0, 2.0),
    },
}
