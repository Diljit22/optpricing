from quantfin.models.merton_jump import MertonJumpModel

MERTON_WORKFLOW_CONFIG = {
    "name": "Merton",
    "model_class": MertonJumpModel,
    "historical_params": ["sigma", "lambda", "mu_j", "sigma_j"],
    "initial_guess": {
        "sigma": 0.18,
        "lambda": 0.2,
        "mu_j": -0.1,
        "sigma_j": 0.15,
        "max_sum_terms": 100,
    },
    # Freeze all jump parameters and the technical term.
    "frozen": ["lambda", "mu_j", "sigma_j", "max_sum_terms"],
    # Only fit sigma to the smile.
    "bounds": {"sigma": (0.01, 1.0)},
}
