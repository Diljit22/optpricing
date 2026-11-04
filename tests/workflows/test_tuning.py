import pandas as pd
import pytest

from optpricing.calibration import CalibrationSettings
from optpricing.data import market_data_manager
from optpricing.models import MertonJumpModel
from optpricing.workflows import (
    CANDIDATE_SETTINGS,
    TuningResult,
    load_tuned_settings,
    load_tuning,
    tune_ticker,
    tuning,
)


@pytest.fixture
def corpus(tmp_path, monkeypatch, synthetic_snapshot):
    """Three days of Merton-generated chains with drifting jump intensity."""
    for date, lam in (("2025-06-16", 0.5), ("2025-06-17", 0.6), ("2025-06-18", 0.7)):
        model = MertonJumpModel(
            params={
                "sigma": 0.15,
                "lambda": lam,
                "mu_j": -0.15,
                "sigma_j": 0.2,
                "max_sum_terms": 100,
            }
        )
        synthetic_snapshot(model, snapshot_date=date).to_parquet(
            tmp_path / f"TEST_{date}.parquet"
        )
    monkeypatch.setattr(market_data_manager, "MARKET_SNAPSHOT_DIR", tmp_path)
    monkeypatch.setattr(tuning, "TUNING_DIR", tmp_path / "tuning")
    return tmp_path


def test_tune_ticker_ranks_models_and_caches_settings(corpus, no_network):
    messages = []
    result = tune_ticker(
        "test",
        models=["BSM", "Merton"],
        candidates=["default", "single-start"],
        progress=messages.append,
    )
    assert result.ticker == "TEST"
    assert result.dates == ["2025-06-16", "2025-06-17", "2025-06-18"]
    assert len(result.trials) == 4 and len(messages) == 4
    assert [row["model"] for row in result.leaderboard] == ["Merton", "BSM"]
    assert result.best_model == "Merton"
    assert result.leaderboard[0]["rank"] == 1
    assert result.leaderboard[0]["oos_iv_rmse"] < result.leaderboard[1]["oos_iv_rmse"]
    assert set(result.settings) == {"BSM", "Merton"}
    assert isinstance(result.settings_for("Merton"), CalibrationSettings)
    frame = result.to_frame()
    assert list(frame["model"]) == ["Merton", "BSM"]

    # cached on disk and reloadable
    path = tuning.tuning_path("TEST")
    assert path.exists()
    loaded = load_tuning("TEST")
    assert loaded.leaderboard == result.leaderboard
    tuned = load_tuned_settings("TEST", "BSM")
    assert tuned is not None and tuned.name in {"default", "single-start"}
    assert load_tuned_settings("TEST", "Heston") is None
    assert load_tuned_settings("NOPE", "BSM") is None


def test_tune_ticker_options_and_errors(corpus, no_network):
    with pytest.raises(ValueError, match="Unknown candidate"):
        tune_ticker("TEST", models=["BSM"], candidates=["nope"])
    with pytest.raises(ValueError, match="at least two"):
        tune_ticker("TEST", models=["BSM"], dates=["2025-06-16"])

    result = tune_ticker(
        "TEST", models=["BSM"], max_pairs=1, candidates="quick", save=False
    )
    assert result.dates == ["2025-06-17", "2025-06-18"]
    assert len(result.trials) == 3  # quick = three candidates
    assert not tuning.tuning_path("TEST").exists()


def test_candidate_settings_are_distinct_and_named():
    names = [s.name for s in CANDIDATE_SETTINGS.values()]
    assert names == list(CANDIDATE_SETTINGS)
    assert len({s.to_dict().__repr__() for s in CANDIDATE_SETTINGS.values()}) == len(
        names
    )


def test_tuning_result_roundtrip(tmp_path):
    result = TuningResult(
        ticker="X",
        created="2025-01-01T00:00:00",
        dates=["a", "b"],
        leaderboard=[
            {"model": "BSM", "settings": "default", "oos_iv_rmse": 0.1, "rank": 1}
        ],
        settings={"BSM": CalibrationSettings().to_dict()},
    )
    path = result.save(tmp_path / "X.json")
    loaded = TuningResult.load(path)
    assert loaded == result
    assert isinstance(loaded.to_frame(), pd.DataFrame)
