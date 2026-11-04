import pytest
from typer.testing import CliRunner

from optpricing.cli.commands import backtest as backtest_cmd
from optpricing.cli.main import app
from optpricing.data import market_data_manager
from optpricing.models import BSMModel
from optpricing.workflows import backtest_workflow, tuning

runner = CliRunner()


@pytest.fixture
def corpus(tmp_path, monkeypatch, synthetic_snapshot):
    for date, vol in (("2025-06-16", 0.2), ("2025-06-17", 0.25), ("2025-06-18", 0.3)):
        synthetic_snapshot(BSMModel({"sigma": vol}), snapshot_date=date).to_parquet(
            tmp_path / f"TEST_{date}.parquet"
        )
    monkeypatch.setattr(market_data_manager, "MARKET_SNAPSHOT_DIR", tmp_path)
    monkeypatch.setattr(backtest_workflow, "BACKTEST_LOGS_DIR", tmp_path / "logs")
    monkeypatch.setattr(tuning, "TUNING_DIR", tmp_path / "tuning")
    return tmp_path


def test_backtest_command_single_model(corpus, no_network):
    result = runner.invoke(
        app, ["backtest", "--ticker", "test", "--model", "BSM", "--n-starts", "1"]
    )
    assert result.exit_code == 0, result.output
    assert "Walk-forward backtest - TEST" in result.output
    assert (corpus / "logs").exists()
    files = sorted(p.name for p in (corpus / "logs").iterdir())
    assert any(f.startswith("TEST_BSM_backtest_") and f.endswith(".csv") for f in files)
    assert any(f.endswith("_buckets.csv") for f in files)
    assert any(f.endswith(".json") for f in files)


def test_backtest_all_models(corpus, no_network, monkeypatch):
    monkeypatch.setattr(
        backtest_cmd, "select_models", lambda model, allow_mc: ["BSM", "Merton"]
    )
    result = runner.invoke(
        app, ["backtest", "-t", "TEST", "-m", "all", "--n-starts", "1"]
    )
    assert result.exit_code == 0, result.output
    assert "Merton" in result.output and "BSM" in result.output


def test_backtest_unknown_model_and_too_few_dates(
    corpus, no_network, tmp_path, monkeypatch
):
    assert runner.invoke(app, ["backtest", "-t", "TEST", "-m", "Nope"]).exit_code == 1
    monkeypatch.setattr(market_data_manager, "MARKET_SNAPSHOT_DIR", tmp_path / "empty")
    (tmp_path / "empty").mkdir()
    result = runner.invoke(app, ["backtest", "-t", "TEST", "-m", "BSM"])
    assert result.exit_code == 0
    assert "Not enough snapshots" in result.output
