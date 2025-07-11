import pandas as pd
import plotly.graph_objects as go
import pytest

from optpricing.dashboard.plots import (
    plot_calendar_heatmap,
    plot_error_heatmap,
    plot_iv_surface_3d,
    plot_smiles_by_expiry,
)


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
            "iv": [0.20, 0.18, 0.22, 0.21],
        }
    )


@pytest.fixture
def mock_market_data():
    """A sample DataFrame for market data with calendar heatmap columns."""
    return pd.DataFrame(
        {
            "expiry": pd.to_datetime(
                ["2023-12-15", "2023-12-15", "2024-01-19", "2024-01-19"]
            ),
            "strike": [100, 105, 100, 105],
            "openInterest": [1000, 1500, 1200, 1800],
            "volume": [500, 600, 700, 800],
            "optionType": ["call", "call", "put", "put"],
        }
    )


def test_plot_smiles_by_expiry_smoke_test(mock_surface_data):
    """
    Smoke test for plot_smiles_by_expiry.
    Ensures the function runs without errors and returns a plotly Figure.
    """
    market_surface = mock_surface_data
    model_surfaces = {"TestModel": mock_surface_data}
    fig = plot_smiles_by_expiry(market_surface, model_surfaces)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 4


def test_plot_smiles_by_expiry_empty_model(mock_surface_data):
    """
    Test plot_smiles_by_expiry with empty model surfaces.
    """
    market_surface = mock_surface_data
    model_surfaces = {}
    fig = plot_smiles_by_expiry(market_surface, model_surfaces)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2


def test_plot_smiles_by_expiry_single_expiry(mock_surface_data):
    """
    Test plot_smiles_by_expiry with a single expiry.
    """
    market_surface = mock_surface_data[mock_surface_data["expiry"] == "2023-12-15"]
    model_surfaces = {"TestModel": market_surface}
    fig = plot_smiles_by_expiry(market_surface, model_surfaces)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2


def test_plot_iv_surface_3d_smoke_test(mock_surface_data):
    """
    Smoke test for plot_iv_surface_3d.
    Ensures the function runs without errors and returns a plotly Figure.
    """
    market_surface = mock_surface_data
    model_surfaces = {"TestModel": mock_surface_data}
    fig = plot_iv_surface_3d(market_surface, model_surfaces)
    assert isinstance(fig, go.Figure)


def test_plot_iv_surface_3d_no_models(mock_surface_data):
    """
    Test plot_iv_surface_3d with no model surfaces.
    """
    market_surface = mock_surface_data
    model_surfaces = {}
    fig = plot_iv_surface_3d(market_surface, model_surfaces)
    assert isinstance(fig, go.Figure)


def test_plot_iv_surface_3d_missing_iv(mock_surface_data):
    """
    Test plot_iv_surface_3d with missing IV values.
    """
    market_surface = mock_surface_data.copy()
    market_surface.loc[0, "iv"] = None
    model_surfaces = {"TestModel": market_surface}
    fig = plot_iv_surface_3d(market_surface, model_surfaces)
    assert isinstance(fig, go.Figure)


def test_plot_error_heatmap(mock_surface_data):
    """
    Test plot_error_heatmap for a given model.
    """
    market_surface = mock_surface_data
    model_surface = mock_surface_data.copy()
    model_surface["iv"] = model_surface["iv"] + 0.01
    fig = plot_error_heatmap(market_surface, model_surface, "TestModel")
    assert isinstance(fig, go.Figure)
    assert "Error" in fig.layout.title.text
    assert fig.data[0].type == "heatmap"


def test_plot_error_heatmap_empty_data(mock_surface_data):
    """
    Test plot_error_heatmap with empty data.
    """
    market_surface = pd.DataFrame(columns=["expiry", "strike", "iv"])
    model_surface = pd.DataFrame(columns=["expiry", "strike", "iv"])
    fig = plot_error_heatmap(market_surface, model_surface, "TestModel")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 0


def test_plot_calendar_heatmap(mock_market_data):
    """
    Test plot_calendar_heatmap with market data.
    """
    market_data = mock_market_data
    fig = plot_calendar_heatmap(market_data)
    assert isinstance(fig, go.Figure)
    assert fig.data[0].type == "heatmap"
    assert "Open Interest" in fig.layout.title.text or "Volume" in fig.layout.title.text


def test_plot_calendar_heatmap_empty_data():
    """
    Test plot_calendar_heatmap with empty data.
    """
    market_data = pd.DataFrame(
        columns=[
            "expiry",
            "strike",
            "openInterest",
            "volume",
        ]
    )
    fig = plot_calendar_heatmap(market_data)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 0
