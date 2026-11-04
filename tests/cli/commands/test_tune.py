import pytest
from typer.testing import CliRunner

from optpricing.cli.main import app
from optpricing.data import market_data_manager
from optpricing.models import BSMModel
from optpricing.workflows import tuning

runner = CliRunner()


@pytest.fixture
def corpus(tmp_path, monkeypatch, synthetic_snapshot):
    for date, vol in (("2025-06-16", 0.2), ("2025-06-17", 0.22)):
        synthetic_snapshot(BSMModel({"sigma": vol}), snapshot_date=date).to_parquet(
            tmp_path / f"TEST_{date}.parquet"
        )
    monkeypatch.setattr(market_data_manager, "MARKET_SNAPSHOT_DIR", tmp_path)
    monkeypatch.setattr(tuning, "TUNING_DIR", tmp_path / "tuning")
    return tmp_path


def test_tune_command(corpus, no_network):
    result = runner.invoke(
        app,
        [
            "tune",
            "--ticker",
            "test",
            "--model",
            "BSM",
            "--candidates",
            "default,single-start",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "tuning BSM / default" in result.output
    assert "Tuning leaderboard - TEST" in result.output
    assert "Best model for TEST: BSM" in result.output
    assert (corpus / "tuning" / "TEST.json").exists()


def test_tune_command_errors(corpus, no_network):
    bad = runner.invoke(
        app, ["tune", "-t", "TEST", "-m", "BSM", "--candidates", "nope"]
    )
    assert bad.exit_code == 1 and "Unknown candidate" in bad.output
    none = runner.invoke(app, ["tune", "-t", "NODATA", "-m", "BSM"])
    assert none.exit_code == 1 and "at least two" in none.output
