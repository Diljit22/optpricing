from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from optpricing.calibration import CalibrationSettings, LiquidityFilter
from optpricing.config import TUNING_DIR
from optpricing.data import get_available_snapshot_dates

from .backtest_workflow import BacktestWorkflow
from .configs import calibratable_model_names, get_model_config

__doc__ = """
Per-ticker auto-tuning by walk-forward cross-validation.

For every model and every candidate :class:`CalibrationSettings`, the ticker's
snapshot corpus is backtested (calibrate on day ``t``, evaluate on ``t+1``);
the settings with the lowest mean out-of-sample IV RMSE win. The result - best
settings per model and a model leaderboard - is cached in
``artifacts/tuning/{ticker}.json`` and picked up by ``optpricing calibrate
--auto``, the backtest and the dashboard, so a plain ticker is all a user has
to provide.
"""

logger = logging.getLogger(__name__)

CANDIDATE_SETTINGS: dict[str, CalibrationSettings] = {
    "default": CalibrationSettings(name="default"),
    "near-the-money": CalibrationSettings(
        name="near-the-money", liquidity=LiquidityFilter(moneyness=(0.85, 1.15))
    ),
    "front-year": CalibrationSettings(
        name="front-year",
        liquidity=LiquidityFilter(min_maturity_days=14, max_maturity_years=1.0),
    ),
    "unweighted": CalibrationSettings(name="unweighted", weighting="none"),
    "price-space": CalibrationSettings(
        name="price-space", objective="price", weighting="vega"
    ),
    "single-start": CalibrationSettings(name="single-start", n_starts=1),
}
QUICK_CANDIDATES = ("default", "near-the-money", "front-year")


@dataclass
class TuningResult:
    """
    Cached outcome of :func:`tune_ticker`.

    Attributes
    ----------
    ticker : str
        Ticker symbol.
    created : str
        ISO timestamp of the run.
    dates : list[str]
        Snapshot dates used (ascending).
    leaderboard : list[dict[str, Any]]
        One row per model, best first: ``model``, ``settings``, ``oos_iv_rmse``,
        ``in_sample_iv_rmse``, ``oos_price_rmse``, ``bsm_atm_iv_rmse``,
        ``runtime_s``, ``n_periods``, ``n_failed``.
    settings : dict[str, dict[str, Any]]
        Best settings (serialised) per model.
    trials : list[dict[str, Any]]
        Every (model, settings) score.
    """

    ticker: str
    created: str
    dates: list[str]
    leaderboard: list[dict[str, Any]] = field(default_factory=list)
    settings: dict[str, dict[str, Any]] = field(default_factory=dict)
    trials: list[dict[str, Any]] = field(default_factory=list)

    @property
    def best_model(self) -> str | None:
        return self.leaderboard[0]["model"] if self.leaderboard else None

    def settings_for(self, model: str) -> CalibrationSettings | None:
        data = self.settings.get(model)
        return CalibrationSettings.from_dict(data) if data else None

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.leaderboard)

    def save(self, path: Path | None = None) -> Path:
        path = path or tuning_path(self.ticker)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2, default=_json_default)
        return path

    @classmethod
    def load(cls, path: Path) -> TuningResult:
        with open(path) as f:
            data = json.load(f)
        return cls(**{k: data[k] for k in cls.__dataclass_fields__ if k in data})


def tuning_path(ticker: str) -> Path:
    """``artifacts/tuning/{TICKER}.json``."""
    return TUNING_DIR / f"{ticker.upper()}.json"


def load_tuning(ticker: str) -> TuningResult | None:
    """The cached tuning result for a ticker, or None."""
    path = tuning_path(ticker)
    if not path.exists():
        return None
    try:
        return TuningResult.load(path)
    except (OSError, ValueError, KeyError) as e:
        logger.warning("Could not read %s: %s", path, e)
        return None


def load_tuned_settings(ticker: str, model: str) -> CalibrationSettings | None:
    """Best cached settings for ``(ticker, model)``, or None when not tuned."""
    result = load_tuning(ticker)
    return result.settings_for(model) if result else None


def _score(summary: dict[str, Any]) -> float:
    value = summary.get("mean_oos_iv_rmse", np.nan)
    return float(value) if value is not None else np.nan


def tune_ticker(
    ticker: str,
    *,
    models: list[str] | None = None,
    candidates: list[str] | str | None = None,
    dates: list[str] | None = None,
    max_pairs: int | None = None,
    allow_mc: bool = False,
    progress: Callable[[str], None] | None = None,
    save: bool = True,
) -> TuningResult:
    """
    Walk-forward tuning of calibration settings and model ranking for a ticker.

    Parameters
    ----------
    ticker : str
        Ticker with at least two stored snapshots.
    models : list[str] | None, optional
        Recipe names; defaults to every calibratable model.
    candidates : list[str] | str | None, optional
        Candidate setting names from :data:`CANDIDATE_SETTINGS`, ``"quick"``
        (three candidates, the default) or ``"full"``.
    dates : list[str] | None, optional
        Snapshot dates to use; defaults to all stored dates.
    max_pairs : int | None, optional
        Use only the most recent ``max_pairs + 1`` dates.
    allow_mc : bool, optional
        Include simulation-only models (slow).
    progress : Callable[[str], None] | None, optional
        Called with a short message before each trial.
    save : bool, optional
        Write the result to ``artifacts/tuning``, by default True.

    Returns
    -------
    TuningResult
        Best settings per model and the leaderboard.

    Raises
    ------
    ValueError
        If fewer than two snapshot dates are available.
    """
    ticker = ticker.upper()
    dates = sorted(dates or get_available_snapshot_dates(ticker, ascending=True))
    if max_pairs is not None:
        dates = dates[-(max_pairs + 1) :]
    if len(dates) < 2:
        raise ValueError(
            f"Tuning {ticker} needs at least two snapshots; found {len(dates)}."
        )
    models = models or calibratable_model_names(allow_mc=allow_mc)
    if candidates is None or candidates == "quick":
        candidate_names = list(QUICK_CANDIDATES)
    elif candidates == "full":
        candidate_names = list(CANDIDATE_SETTINGS)
    else:
        candidate_names = list(candidates)
    unknown = [c for c in candidate_names if c not in CANDIDATE_SETTINGS]
    if unknown:
        raise ValueError(f"Unknown candidate settings: {unknown}")

    result = TuningResult(
        ticker=ticker,
        created=pd.Timestamp.now().isoformat(timespec="seconds"),
        dates=dates,
    )
    for model in models:
        cfg = get_model_config(model)
        best: dict[str, Any] | None = None
        for name in candidate_names:
            settings = CANDIDATE_SETTINGS[name]
            if progress:
                progress(f"{model} / {name}")
            t0 = time.perf_counter()
            bt = BacktestWorkflow(
                ticker, cfg, settings=settings, dates=dates, allow_mc=allow_mc
            )
            bt.run()
            summary = bt.summary()
            trial = {
                "model": cfg["name"],
                "settings": name,
                "oos_iv_rmse": _score(summary),
                "in_sample_iv_rmse": summary.get("mean_in_sample_iv_rmse", np.nan),
                "oos_price_rmse": summary.get("mean_oos_price_rmse", np.nan),
                "bsm_atm_iv_rmse": summary.get("mean_bsm_atm_iv_rmse", np.nan),
                "runtime_s": time.perf_counter() - t0,
                "n_periods": summary.get("n_periods", 0),
                "n_failed": summary.get("n_failed", 0),
            }
            result.trials.append(trial)
            logger.info(
                "%s / %s: OOS IV RMSE %.4f (%d periods, %.1fs)",
                cfg["name"],
                name,
                trial["oos_iv_rmse"],
                trial["n_periods"],
                trial["runtime_s"],
            )
            if np.isfinite(trial["oos_iv_rmse"]) and (
                best is None
                or trial["oos_iv_rmse"] < best["oos_iv_rmse"]
                or (
                    trial["oos_iv_rmse"] == best["oos_iv_rmse"]
                    and trial["runtime_s"] < best["runtime_s"]
                )
            ):
                best = trial
        if best is None:
            logger.warning(
                "No successful trial for %s; left out of the leaderboard.", model
            )
            continue
        result.leaderboard.append(dict(best))
        result.settings[cfg["name"]] = CANDIDATE_SETTINGS[best["settings"]].to_dict()

    result.leaderboard.sort(key=lambda row: row["oos_iv_rmse"])
    for rank, row in enumerate(result.leaderboard, start=1):
        row["rank"] = rank
    if save and result.leaderboard:
        path = result.save()
        logger.info("Tuning result saved to %s", path)
    return result


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.integer | np.floating):
        value = obj.item()
        return None if isinstance(value, float) and not np.isfinite(value) else value
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return str(obj)
