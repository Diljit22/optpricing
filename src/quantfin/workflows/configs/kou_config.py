from quantfin.models.kou import KouModel

KOU_WORKFLOW_CONFIG = {
    "model_name": "Kou",
    "model_class": KouModel,
    "historical_params": ["sigma", "lambda"],
    
    "initial_guess": {'sigma': 0.15, 'lambda': 1.0, 'p_up': 0.6, 'eta1': 10.0, 'eta2': 5.0},
    
    "frozen": ['lambda', 'p_up', 'eta1', 'eta2'],
    
    "bounds": {'sigma': (0.01, 1.0)}
}
