import numpy as np
import pandas as pd
import pytest

from optpricing.calibration.fit_jump_parameters import fit_jump_params_from_history


def test_fit_jump_params_with_three_jumps():
    """
    Tests jump parameter estimation when there are three out of threshold returns.
    """

    # 250 diffusion days at +-1%
    diffusion = np.array([-0.01, 0.01] * 125)
    jumps = np.array([0.05, -0.06, 0.07])
    log_returns = pd.Series(np.concatenate([diffusion, jumps]))

    threshold_stds = 2.0
    params = fit_jump_params_from_history(log_returns, threshold_stds=threshold_stds)

    diffusion_returns = log_returns[
        log_returns.abs() < threshold_stds * log_returns.std()
    ]
    jump_returns = log_returns[log_returns.abs() >= threshold_stds * log_returns.std()]

    expected_sigma = diffusion_returns.std() * np.sqrt(252)
    expected_lambda = len(jump_returns) / len(log_returns) * 252
    expected_mu_j = jump_returns.mean()
    expected_sigma_j = jump_returns.std()

    # Allow a tiny relative tolerance for floating‐point
    assert params["sigma"] == pytest.approx(expected_sigma, rel=1e-3)
    assert params["lambda"] == pytest.approx(expected_lambda, rel=1e-3)
    assert params["mu_j"] == pytest.approx(expected_mu_j, abs=1e-8)
    assert params["sigma_j"] == pytest.approx(expected_sigma_j, rel=1e-3)
