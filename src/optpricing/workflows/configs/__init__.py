from __future__ import annotations

__doc__ = """
This package contains pre-defined "recipes" for calibrating each financial model.

Each config is a dictionary that specifies the model class, initial parameter
guesses, bounds for optimization, and any special handling such as a
historical-returns prior for jump parameters (``historical_prior``) or the
need for Monte Carlo pricing (``requires_mc``).
"""

from typing import Any

from optpricing.calibration.vectorized_pricer import supports_vectorized_pricing

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

ALL_MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "BSM": BSM_WORKFLOW_CONFIG,
    "Merton": MERTON_WORKFLOW_CONFIG,
    "Heston": HESTON_WORKFLOW_CONFIG,
    "Bates": BATES_WORKFLOW_CONFIG,
    "SABR": SABR_WORKFLOW_CONFIG,
    "SABRJump": SABR_JUMP_WORKFLOW_CONFIG,
    "CEV": CEV_WORKFLOW_CONFIG,
    "Kou": KOU_WORKFLOW_CONFIG,
    "NIG": NIG_WORKFLOW_CONFIG,
    "VG": VG_WORKFLOW_CONFIG,
    "CGMY": CGMY_WORKFLOW_CONFIG,
    "Hyperbolic": HYPERBOLIC_WORKFLOW_CONFIG,
}

# Alternative spellings accepted on the command line and in the dashboard.
MODEL_ALIASES: dict[str, str] = {
    "variance gamma": "VG",
    "variancegamma": "VG",
    "variance_gamma": "VG",
    "black-scholes": "BSM",
    "black-scholes-merton": "BSM",
    "blackscholes": "BSM",
    "merton jump-diffusion": "Merton",
    "mertonjump": "Merton",
    "merton_jump": "Merton",
    "sabr with jumps": "SABRJump",
    "sabr_jump": "SABRJump",
    "sabr-jump": "SABRJump",
    "kou double-exponential jump": "Kou",
    "normal inverse gaussian": "NIG",
}


def resolve_model_name(name: str) -> str:
    """
    Canonical recipe key for a user-supplied model name (case-insensitive,
    aliases such as ``"Variance Gamma"`` accepted).

    Raises
    ------
    KeyError
        If the name matches no recipe.
    """
    key = name.strip()
    if key in ALL_MODEL_CONFIGS:
        return key
    lowered = key.lower()
    for canonical in ALL_MODEL_CONFIGS:
        if canonical.lower() == lowered:
            return canonical
    if lowered in MODEL_ALIASES:
        return MODEL_ALIASES[lowered]
    for canonical, cfg in ALL_MODEL_CONFIGS.items():
        if cfg["model_class"].name.lower() == lowered:
            return canonical
    raise KeyError(
        f"Unknown model '{name}'. Available: {', '.join(ALL_MODEL_CONFIGS)}."
    )


def get_model_config(name: str) -> dict[str, Any]:
    """A copy of the recipe for ``name`` (see :func:`resolve_model_name`)."""
    return dict(ALL_MODEL_CONFIGS[resolve_model_name(name)])


def calibratable_model_names(*, allow_mc: bool = False) -> list[str]:
    """
    Recipe keys whose model can be fitted to a chain: models with a vectorised
    closed form or a characteristic function, plus (with ``allow_mc``) the
    simulation-only ones.
    """
    names = []
    for key, cfg in ALL_MODEL_CONFIGS.items():
        model = cfg["model_class"](params=dict(cfg["model_class"].default_params))
        if supports_vectorized_pricing(model, allow_mc=allow_mc):
            names.append(key)
    return names


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
    "MODEL_ALIASES",
    "NIG_WORKFLOW_CONFIG",
    "SABR_WORKFLOW_CONFIG",
    "SABR_JUMP_WORKFLOW_CONFIG",
    "VG_WORKFLOW_CONFIG",
    "calibratable_model_names",
    "get_model_config",
    "resolve_model_name",
]
