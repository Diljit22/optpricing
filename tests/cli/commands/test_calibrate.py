import json

import pytest
from typer.testing import CliRunner

from optpricing.calibration import CalibrationSettings
from optpricing.cli.commands import calibrate as calibrate_cmd
from optpricing.cli.main import app
from optpricing.data import market_data_manager
from optpricing.models import BSMModel
from optpricing.workflows import TuningResult, tuning

runner = CliRunner()


@pytest.fixture
def store(tmp_path, monkeypatch, synthetic_snapshot):
    """One BSM snapshot on disk plus redirected artifact directories."""
    df = synthetic_snapshot(BSMModel({"sigma": 0.23}), snapshot_date="2025-06-16")
    df.to_parquet(tmp_path / "TEST_2025-06-16.parquet")
    monkeypatch.setattr(market_data_manager, "MARKET_SNAPSHOT_DIR", tmp_path)
    monkeypatch.setattr(calibrate_cmd, "CALIBRATED_PARAMS_DIR", tmp_path / "params")
    monkeypatch.setattr(calibrate_cmd, "CALIBRATION_LOGS_DIR", tmp_path / "logs")
    monkeypatch.setattr(tuning, "TUNING_DIR", tmp_path / "tuning")
    return tmp_path


def test_calibrate_single_model_latest_date(store, no_network):
    result = runner.invoke(
        app, ["calibrate", "--ticker", "test", "--model", "BSM", "--n-starts", "1"]
    )
    assert result.exit_code == 0, result.output
    assert "Using latest date: 2025-06-16" in result.output
    assert "Calibration leaderboard" in result.output
    saved = json.loads((store / "params" / "TEST_BSM_2025-06-16.json").read_text())
    assert saved["params"]["sigma"] == pytest.approx(0.23, abs=1e-4)
    assert saved["rate"] == pytest.approx(0.04)
    assert saved["settings"]["n_starts"] == 1
    assert (store / "logs" / "TEST_leaderboard_2025-06-16.csv").exists()


def test_calibrate_all_models_ranks_them(store, no_network, monkeypatch):
    # keep the run short: pretend only two models are calibratable
    monkeypatch.setattr(
        calibrate_cmd,
        "calibratable_model_names",
        lambda allow_mc=False: ["Merton", "BSM"],
    )
    result = runner.invoke(
        app,
        ["calibrate", "-t", "TEST", "-m", "all", "-d", "2025-06-16", "--n-starts", "1"],
    )
    assert result.exit_code == 0, result.output
    lines = [line for line in result.output.splitlines() if "│" in line]
    # BSM generated the data, so it must rank first
    assert any("BSM" in line for line in lines[:3])
    assert (store / "params" / "TEST_Merton_2025-06-16.json").exists()


def test_calibrate_with_fixed_param_alias_and_objective(store, no_network):
    result = runner.invoke(
        app,
        [
            "calibrate",
            "-t",
            "TEST",
            "-m",
            "black-scholes",
            "-d",
            "2025-06-16",
            "--fix",
            "sigma=0.3",
            "--objective",
            "price",
        ],
    )
    assert result.exit_code == 0, result.output
    saved = json.loads((store / "params" / "TEST_BSM_2025-06-16.json").read_text())
    assert saved["params"] == {"sigma": 0.3}


def test_calibrate_auto_uses_tuned_settings(store, no_network):
    tuned = CalibrationSettings(name="near-the-money", n_starts=1)
    TuningResult(
        ticker="TEST",
        created="x",
        dates=["a", "b"],
        leaderboard=[
            {
                "model": "BSM",
                "settings": "near-the-money",
                "oos_iv_rmse": 0.1,
                "rank": 1,
            }
        ],
        settings={"BSM": tuned.to_dict()},
    ).save()
    result = runner.invoke(app, ["calibrate", "-t", "TEST", "-m", "BSM", "--auto"])
    assert result.exit_code == 0, result.output
    assert "near-the-money settings" in result.output
    missing = runner.invoke(
        app, ["calibrate", "-t", "TEST", "-m", "Heston", "--auto", "--n-starts", "1"]
    )
    assert "No tuned settings for TEST/Heston" in missing.output


def test_calibrate_errors(store, no_network):
    assert runner.invoke(app, ["calibrate", "-t", "NOPE", "-m", "BSM"]).exit_code == 1
    bad_fix = runner.invoke(
        app, ["calibrate", "-t", "TEST", "-m", "BSM", "--fix", "sigma"]
    )
    assert bad_fix.exit_code == 1 and "Invalid format" in bad_fix.output
    unknown = runner.invoke(app, ["calibrate", "-t", "TEST", "-m", "NotAModel"])
    assert unknown.exit_code == 1 and "Unknown model" in unknown.output
    bad_obj = runner.invoke(
        app, ["calibrate", "-t", "TEST", "-m", "BSM", "--objective", "x"]
    )
    assert bad_obj.exit_code == 1
    mc_only = runner.invoke(app, ["calibrate", "-t", "TEST", "-m", "SABRJump"])
    assert mc_only.exit_code == 1 and "FAILED" in mc_only.output
