from __future__ import annotations

__doc__ = """
This package contains all components for the Streamlit dashboard UI,
including the main service layer, plotting functions, and UI widgets.
"""

from .service import DashboardService
from .widgets import render_parity_analysis_widget

__all__ = [
    "DashboardService",
    "render_parity_analysis_widget",
]
