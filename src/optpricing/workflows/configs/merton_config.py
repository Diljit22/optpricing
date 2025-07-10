from optpricing.models.merton_jump import MertonJumpModel

__doc__ = """
Calibration recipe for the Merton-Jump model.
"""

MERTON_WORKFLOW_CONFIG = {
    "name": "Merton",
    "model_class": MertonJumpModel,
    "use_historical_strategy": True,
    "frozen": {},
}
