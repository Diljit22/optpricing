from quantfin.models.bsm import BSMModel

BSM_WORKFLOW_CONFIG = {
    "name": "BSM",
    "model_class": BSMModel,
    "initial_guess": {'sigma': 0.20},
    "frozen": {},
    "bounds": {'sigma': (0.01, 1.0)}
}