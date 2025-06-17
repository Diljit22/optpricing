import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from typing import List, Dict, Any
import numpy as np

def plot_smiles_by_expiry(market_surface: pd.DataFrame, model_surfaces: Dict[str, pd.DataFrame], ticker: str, snapshot_date: str):
    expiries = sorted(market_surface['expiry'].unique())
    plot_expiries = [expiries[i] for i in np.linspace(0, len(expiries)-1, min(4, len(expiries)), dtype=int)]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharey=True)
    fig.suptitle(f'Volatility Smiles for {ticker} on {snapshot_date}', fontsize=18)
    axes = axes.flatten()

    for i, expiry in enumerate(plot_expiries):
        ax = axes[i]
        market_slice = market_surface[market_surface['expiry'] == expiry]
        
        ax.scatter(market_slice['strike'], market_slice['iv'] * 100, label='Market IV', alpha=0.6, s=15, zorder=5)
        
        for model_name, model_surface in model_surfaces.items():
            model_slice = model_surface[model_surface['expiry'] == expiry].sort_values('strike')
            ax.plot(model_slice['strike'], model_slice['iv'] * 100, label=f'{model_name} IV', lw=2.5)
            
        expiry_date_str = pd.to_datetime(expiry).strftime('%Y-%m-%d')
        maturity = market_slice['maturity'].iloc[0]
        ax.set_title(f"Expiry: {expiry_date_str} (T={maturity:.2f}y)")
        ax.set_xlabel("Strike")
        ax.set_ylabel("Implied Volatility (%)")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
        # Set sensible y-axis limits to keep the plot readable
        ax.set_ylim(bottom=0, top=100) # IVs are rarely outside 0-100%

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig
def plot_iv_surface(
    market_surface: pd.DataFrame,
    model_surfaces: Dict[str, pd.DataFrame],
    ticker: str
):
    """
    Creates an interactive 3D plot of the market and model volatility surfaces.
    """
    fig = go.Figure()

    # Add Market Surface
    fig.add_trace(go.Mesh3d(
        x=market_surface['maturity'], y=market_surface['strike'], z=market_surface['iv'],
        opacity=0.5, color='grey', name='Market Surface'
    ))

    # Add Model Surfaces
    colors = ['blue', 'red', 'green']
    for i, (model_name, model_surface) in enumerate(model_surfaces.items()):
        fig.add_trace(go.Mesh3d(
            x=model_surface['maturity'], y=model_surface['strike'], z=model_surface['iv'],
            opacity=0.5, color=colors[i % len(colors)], name=f'{model_name} Surface'
        ))

    fig.update_layout(
        title=f'Implied Volatility Surface for {ticker}',
        scene=dict(
            xaxis_title='Maturity (Years)',
            yaxis_title='Strike',
            zaxis_title='Implied Volatility'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    #fig.show()
    fig.update_layout(...)
    return fig