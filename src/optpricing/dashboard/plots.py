from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import griddata

__doc__ = """
Contains plotting functions used by the dashboard to visualize
volatility surfaces and smiles.
"""


def plot_smiles_by_expiry(
    market_surface: pd.DataFrame,
    model_surfaces: dict[str, pd.DataFrame],
) -> go.Figure:
    """
    Generates a robust Plotly figure with volatility smiles for key expiries.
    """
    if market_surface.empty:
        return go.Figure().update_layout(title="No market data to plot smiles.")

    expiries = sorted(market_surface["expiry"].unique())
    plot_expiries = expiries[:4]

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            f"Expiry: {pd.to_datetime(e).strftime('%Y-%m-%d')}" for e in plot_expiries
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.05,
    )

    for i, expiry in enumerate(plot_expiries):
        row, col = i // 2 + 1, i % 2 + 1
        market_slice = market_surface[market_surface["expiry"] == expiry]

        fig.add_trace(
            go.Scatter(
                x=market_slice["strike"],
                y=market_slice["iv"],
                mode="markers",
                name="Market IV",
                legendgroup="market",
                showlegend=(i == 0),
                marker=dict(symbol="cross", size=8, color="black"),
            ),
            row=row,
            col=col,
        )

        for model_name, model_surface in model_surfaces.items():
            if model_surface.empty:
                continue
            model_slice = model_surface[model_surface["expiry"] == expiry].sort_values(
                "strike"
            )
            fig.add_trace(
                go.Scatter(
                    x=model_slice["strike"],
                    y=model_slice["iv"],
                    mode="lines",
                    name=f"{model_name} IV",
                    legendgroup=model_name,
                    showlegend=(i == 0),
                    line=dict(width=3),
                ),
                row=row,
                col=col,
            )

    # Hide any unused subplots
    for i in range(len(plot_expiries), 4):
        row, col = i // 2 + 1, i % 2 + 1
        fig.update_xaxes(visible=False, row=row, col=col)
        fig.update_yaxes(visible=False, row=row, col=col)

    fig.update_layout(
        height=700,
        title_text="<b>Volatility Smiles by Expiration</b>",
        legend_title_text="Models",
        margin=dict(l=40, r=20, t=60, b=40),
    )
    fig.update_yaxes(title_text="Implied Volatility", tickformat=".2%")
    fig.update_xaxes(title_text="Strike")
    return fig


def plot_iv_surface_3d(
    market_surface: pd.DataFrame,
    model_surfaces: dict[str, pd.DataFrame],
) -> go.Figure:
    """
    Creates an interactive, INTERPOLATED 3D plot of the volatility surfaces.
    """
    fig = go.Figure()

    if market_surface.empty:
        return fig.update_layout(title="No market data for 3D surface.")

    # regular grid to interpolate onto
    grid_t, grid_k = np.mgrid[
        market_surface["maturity"].min() : market_surface["maturity"].max() : 100j,
        market_surface["strike"].min() : market_surface["strike"].max() : 100j,
    ]

    market_points = market_surface[["maturity", "strike"]].values
    market_values = market_surface["iv"].values
    grid_z_market = griddata(
        market_points, market_values, (grid_t, grid_k), method="cubic"
    )

    fig.add_trace(
        go.Surface(
            z=grid_z_market,
            x=grid_t,
            y=grid_k,
            name="Market IV",
            colorscale="Viridis",
            opacity=0.7,
            showscale=False,
        )
    )

    # Interpolate model data
    for name, surface in model_surfaces.items():
        if not surface.empty:
            model_points = surface[["maturity", "strike"]].values
            model_values = surface["iv"].values
            grid_z_model = griddata(
                model_points, model_values, (grid_t, grid_k), method="cubic"
            )
            fig.add_trace(
                go.Surface(
                    z=grid_z_model,
                    x=grid_t,
                    y=grid_k,
                    name=f"{name} IV",
                    opacity=0.9,
                    showscale=False,
                )
            )

    fig.update_layout(
        title="<b>Interpolated Implied Volatility Surface</b>",
        scene=dict(
            xaxis_title="Maturity (Years)",
            yaxis_title="Strike",
            zaxis_title="Implied Volatility",
            zaxis=dict(tickformat=".2%"),
        ),
        height=700,
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(yanchor="top", y=0.9, xanchor="left", x=0.01),
    )
    return fig


def plot_error_heatmap(
    market_surface: pd.DataFrame,
    model_surface: pd.DataFrame,
    model_name: str,
) -> go.Figure:
    """
    Plots a heatmap of the pricing error (Model IV - Market IV) in basis points.
    FIXED: Handles duplicate index entries before pivoting.
    """
    if market_surface.empty or model_surface.empty:
        return go.Figure().update_layout(
            title=f"Not enough data to plot error for {model_name}."
        )

    market_dedup = market_surface.drop_duplicates(subset=["expiry", "strike"])
    model_dedup = model_surface.drop_duplicates(subset=["expiry", "strike"])

    merged = pd.merge(
        market_dedup,
        model_dedup,
        on=["expiry", "strike"],
        suffixes=("_market", "_model"),
    )
    merged["error"] = (
        merged["iv_model"] - merged["iv_market"]
    ) * 10000  # Convert to bps

    pivot_df = merged.pivot(index="expiry", columns="strike", values="error")

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot_df.values,
            x=pivot_df.columns,
            y=[pd.to_datetime(d).strftime("%Y-%m-%d") for d in pivot_df.index],
            colorscale="RdBu",
            zmid=0,
            hovertemplate="Strike: %{x}<br>Expiry: %{y}<br>Error: %{z:.2f} bps<extra></extra>",  # noqa E501
        )
    )

    fig.update_layout(
        title=f"<b>{model_name} vs. Market IV Error</b> (in basis points)",
        xaxis_title="Strike",
        yaxis_title="Expiry",
        height=600,
    )
    return fig


def plot_calendar_heatmap(market_data: pd.DataFrame) -> go.Figure:
    if market_data.empty:
        return go.Figure().update_layout(title="No market data to plot calendar.")
    df = market_data.copy()
    df["expiry_date"] = pd.to_datetime(df["expiry"]).dt.date
    volume_pivot = df.pivot_table(
        index="expiry_date", columns="strike", values="volume", aggfunc="sum"
    )
    fig = go.Figure(
        data=go.Heatmap(
            z=volume_pivot.values,
            x=volume_pivot.columns,
            y=[d.strftime("%Y-%m-%d") for d in volume_pivot.index],
            colorscale="Viridis",
            hovertemplate="Strike: %{x}<br>Expiry: %{y}<br>Total Volume: %{z:,.0f}<extra></extra>",  # noqa E501
        )
    )
    fig.update_layout(
        title="<b>Option Volume Calendar</b>",
        xaxis_title="Strike",
        yaxis_title="Expiry",
        height=600,
    )
    return fig
