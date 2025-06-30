from optpricing.models.merton_jump import MertonJumpModel

MERTON_WORKFLOW_CONFIG = {
    "name": "Merton",
    "model_class": MertonJumpModel,
    "use_historical_strategy": True,
    "frozen": ["lambda", "mu_j", "sigma_j"],
}
