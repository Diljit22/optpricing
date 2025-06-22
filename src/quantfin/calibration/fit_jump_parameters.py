import numpy as np
import pandas as pd


def fit_jump_params_from_history(
    log_returns: pd.Series, threshold_stds: float = 3.0
) -> dict:
    """
    Estimates jump parameters and diffusion volatility from historical returns.
    """
    print("Fitting jump parameters from historical returns...")

    std_dev = log_returns.std()
    jump_threshold = threshold_stds * std_dev

    diffusion_returns = log_returns[abs(log_returns) < jump_threshold]
    jump_returns = log_returns[abs(log_returns) >= jump_threshold]

    # Annualize daily std dev by multiplying by sqrt(252 trading days)
    sigma_est = diffusion_returns.std() * np.sqrt(252)

    if len(jump_returns) > 2:
        lambda_est = len(jump_returns) / len(log_returns) * 252
        mu_j_est = jump_returns.mean()
        sigma_j_est = jump_returns.std()
    else:
        lambda_est, mu_j_est, sigma_j_est = 0.1, 0.0, 0.0
        print("  -> Warning: Not enough jumps detected. Using default jump parameters.")

    # Use the key 'sigma' to match the parameter name in the Merton/Kou models.
    fitted_params = {
        "sigma": sigma_est,
        "lambda": lambda_est,
        "mu_j": mu_j_est,
        "sigma_j": sigma_j_est,
    }

    print(
        f"  -> Estimated Historical Params: { {k: f'{v:.4f}' for k, v in fitted_params.items()} }"
    )
    return fitted_params
