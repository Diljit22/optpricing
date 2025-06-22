import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go


def plot_smiles_by_expiry(
    market_surface: pd.DataFrame,
    model_surfaces: dict[str, pd.DataFrame],
    ticker: str,
    snapshot_date: str,
):
    """Generates a matplotlib figure with volatility smiles for key expiries."""
    expiries = sorted(market_surface["expiry"].unique())
    # Select up to 4 expiries to plot
    plot_indices = np.linspace(0, len(expiries) - 1, min(4, len(expiries)), dtype=int)
    plot_expiries = [expiries[i] for i in plot_indices]

    fig, axes = plt.subplots(2, 2, figsize=(15, 11), sharey=True)
    fig.suptitle(f"Volatility Smiles for {ticker} on {snapshot_date}", fontsize=18)
    axes = axes.flatten()

    for i, expiry in enumerate(plot_expiries):
        ax = axes[i]
        market_slice = market_surface[market_surface["expiry"] == expiry]

        ax.scatter(
            market_slice["strike"],
            market_slice["iv"] * 100,
            label="Market IV",
            alpha=0.7,
            s=20,
            zorder=10,
        )

        for model_name, model_surface in model_surfaces.items():
            model_slice = model_surface[model_surface["expiry"] == expiry].sort_values(
                "strike"
            )
            ax.plot(
                model_slice["strike"],
                model_slice["iv"] * 100,
                label=f"{model_name} IV",
                lw=2.5,
                alpha=0.8,
            )

        expiry_date_str = pd.to_datetime(expiry).strftime("%Y-%m-%d")
        maturity = market_slice["maturity"].iloc[0]
        ax.set_title(f"Expiry: {expiry_date_str} (T={maturity:.2f}y)")
        ax.set_xlabel("Strike")
        ax.set_ylabel("Implied Volatility (%)")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.set_ylim(bottom=max(0, market_slice["iv"].min() * 100 - 5))

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig


def plot_iv_surface_3d(
    market_surface: pd.DataFrame, model_surfaces: dict[str, pd.DataFrame], ticker: str
):
    """Creates an interactive 3D plot of the volatility surfaces."""
    fig = go.Figure()
    fig.add_trace(
        go.Mesh3d(
            x=market_surface["maturity"],
            y=market_surface["strike"],
            z=market_surface["iv"],
            opacity=0.5,
            color="grey",
            name="Market IV",
        )
    )

    colors = [
        "#3a7fc1",  # Deep ocean blue
        "#ff8c00",  # Sunset orange
        "#4aac26",  # Fresh sprout green
        "#d03530",  # Warning red
    ]

    for i, (name, surface) in enumerate(model_surfaces.items()):
        fig.add_trace(
            go.Scatter3d(
                x=surface["maturity"],
                y=surface["strike"],
                z=surface["iv"],
                mode="lines",
                name=f"{name} IV",
                line=dict(color=colors[i % len(colors)], width=6),
            )
        )

    fig.update_layout(
        title=f"Implied Volatility Surface for {ticker}",
        scene=dict(
            xaxis_title="Maturity (Years)",
            yaxis_title="Strike",
            zaxis_title="Implied Volatility",
        ),
        margin=dict(l=0, r=0, b=0, t=40),
    )
    return fig
