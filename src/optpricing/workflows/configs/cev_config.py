from optpricing.models import CEVModel

__doc__ = """
Calibration recipe for the Constant Elasticity of Variance (CEV) model
(priced with the Cox/Schroder closed form).

The local volatility at the spot is ``sigma * S^(gamma - 1)``, so the natural
size of ``sigma`` spans orders of magnitude across ``gamma``; it is therefore
optimised on a log scale over a wide range, and the workflow seeds it with
``ATM vol * S^(1 - gamma)``.
"""

CEV_WORKFLOW_CONFIG = {
    "name": "CEV",
    "model_class": CEVModel,
    "initial_guess": CEVModel.default_params,
    "frozen": {},
    "bounds": {
        "sigma": (1e-3, 1e3),
        "gamma": (0.0, 1.5),
    },
    "log_scale": ["sigma"],
}
