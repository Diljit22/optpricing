from __future__ import annotations

__doc__ = """
This package contains pre-defined "recipes" for calibrating each financial model.

Each config is a dictionary that specifies the model class, initial parameter
guesses, bounds for optimization, and any special handling like freezing
parameters or using historical data.
"""

from .bates_config import BATES_WORKFLOW_CONFIG
from .bsm_config import BSM_WORKFLOW_CONFIG
from .cev_config import CEV_WORKFLOW_CONFIG
from .cgmy_config import CGMY_WORKFLOW_CONFIG
from .heston_config import HESTON_WORKFLOW_CONFIG
from .hyperbolic_config import HYPERBOLIC_WORKFLOW_CONFIG
from .kou_config import KOU_WORKFLOW_CONFIG
from .merton_config import MERTON_WORKFLOW_CONFIG
from .nig_config import NIG_WORKFLOW_CONFIG
from .sabr_config import SABR_WORKFLOW_CONFIG
from .sabr_jump_config import SABR_JUMP_WORKFLOW_CONFIG
from .vg_config import VG_WORKFLOW_CONFIG

ALL_MODEL_CONFIGS = {
    "BSM": BSM_WORKFLOW_CONFIG,
    "Merton": MERTON_WORKFLOW_CONFIG,
    "Heston": HESTON_WORKFLOW_CONFIG,
    "Bates": BATES_WORKFLOW_CONFIG,
    "SABR": SABR_WORKFLOW_CONFIG,
    "SABRJump": SABR_JUMP_WORKFLOW_CONFIG,
    "CEV": CEV_WORKFLOW_CONFIG,
    "Kou": KOU_WORKFLOW_CONFIG,
    "NIG": NIG_WORKFLOW_CONFIG,
    "Variance Gamma": VG_WORKFLOW_CONFIG,
    "CGMY": CGMY_WORKFLOW_CONFIG,
    "Hyperbolic": HYPERBOLIC_WORKFLOW_CONFIG,
}

__all__ = [
    "ALL_MODEL_CONFIGS",
    "BATES_WORKFLOW_CONFIG",
    "BSM_WORKFLOW_CONFIG",
    "CEV_WORKFLOW_CONFIG",
    "CGMY_WORKFLOW_CONFIG",
    "HESTON_WORKFLOW_CONFIG",
    "HYPERBOLIC_WORKFLOW_CONFIG",
    "KOU_WORKFLOW_CONFIG",
    "MERTON_WORKFLOW_CONFIG",
    "NIG_WORKFLOW_CONFIG",
    "SABR_WORKFLOW_CONFIG",
    "SABR_JUMP_WORKFLOW_CONFIG",
    "VG_WORKFLOW_CONFIG",
]
