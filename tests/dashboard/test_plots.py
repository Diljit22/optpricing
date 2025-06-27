import matplotlib.figure
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import pytest

from optpricing.dashboard.plots import plot_iv_surface_3d, plot_smiles_by_expiry


@pytest.fixture
def mock_surface_data():
    """A sample DataFrame that mimics a volatility surface."""
    return pd.DataFrame(
        {
            "expiry": pd.to_datetime(
                ["2023-12-15", "2023-12-15", "2024-01-19", "2024-01-19"]
            ),
            "maturity": [0.1, 0.1, 0.2, 0.2],
            "strike": [100, 105, 100, 105],
            "iv": [0.2, 0.18, 0.22, 0.21],
        }
    )


def test_plot_smiles_by_expiry_smoke_test(mock_surface_data):
    """
    Smoke test for plot_smiles_by_expiry.
    Ensures the function runs without errors and returns a matplotlib Figure.
    """
    market_surface = mock_surface_data
    model_surfaces = {"TestModel": mock_surface_data}

    # The test passes if this call completes without raising an exception
    fig = plot_smiles_by_expiry(market_surface, model_surfaces, "TEST", "2023-11-15")

    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)  # Close the figure to avoid displaying it during tests


def test_plot_iv_surface_3d_smoke_test(mock_surface_data):
    """
    Smoke test for plot_iv_surface_3d.
    Ensures the function runs without errors and returns a plotly Figure.
    """
    market_surface = mock_surface_data
    model_surfaces = {"TestModel": mock_surface_data}

    fig = plot_iv_surface_3d(market_surface, model_surfaces, "TEST")

    assert isinstance(fig, go.Figure)
