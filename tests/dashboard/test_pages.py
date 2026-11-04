"""
Headless smoke tests: every Streamlit page renders against a stored snapshot
without raising, and the calibration page runs a real (fast) calibration.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from optpricing.data import market_data_manager  # noqa: E402
from optpricing.models import BSMModel  # noqa: E402
from optpricing.workflows import tuning  # noqa: E402

PAGES_DIR = Path(__file__).resolve().parents[2] / "src" / "optpricing" / "dashboard"
PAGE_FILES = sorted(
    p for p in (PAGES_DIR / "pages").glob("*.py") if p.stem != "__init__"
)
DATE = "2025-06-16"


@pytest.fixture
def snapshot_store(tmp_path, monkeypatch, synthetic_snapshot):
    """A SPY snapshot (SPY is the sidebar's default ticker) in a temp data dir."""
    df = synthetic_snapshot(BSMModel({"sigma": 0.23}), ticker="SPY", snapshot_date=DATE)
    df.to_parquet(tmp_path / f"SPY_{DATE}.parquet")
    monkeypatch.setattr(market_data_manager, "MARKET_SNAPSHOT_DIR", tmp_path)
    monkeypatch.setattr(tuning, "TUNING_DIR", tmp_path / "tuning")
    return tmp_path


def _run(path: Path, **kwargs) -> AppTest:
    at = AppTest.from_file(str(path), default_timeout=120)
    at.run(**kwargs)
    return at


def test_home_page_renders():
    at = _run(PAGES_DIR / "Home.py")
    assert not at.exception
    assert at.title[0].value.startswith("Welcome")


@pytest.mark.parametrize("page", PAGE_FILES, ids=[p.stem for p in PAGE_FILES])
def test_every_page_renders_with_snapshot_data(page, snapshot_store, no_network):
    at = _run(page)
    assert not at.exception, [e.message for e in at.exception]
    # the shared sidebar lists the stored snapshot date next to "Live Data"
    if at.sidebar.selectbox:
        dates = at.sidebar.selectbox(key="date_selector").options
        assert dates == ["Live Data", DATE]


def test_market_analytics_page_shows_snapshot(snapshot_store, no_network):
    at = _run(PAGES_DIR / "pages" / "3_Market_Analytics.py")
    at.sidebar.selectbox(key="date_selector").select(DATE).run()
    assert not at.exception, [e.message for e in at.exception]
    assert any("Option Chain Explorer" in h.value for h in at.header)
    assert at.dataframe  # raw data tab rendered a table


def test_calibration_page_runs_a_calibration(snapshot_store, no_network):
    page = PAGES_DIR / "pages" / "2_Calibration_and_IV.py"
    at = _run(page)
    at.sidebar.selectbox(key="date_selector").select(DATE).run()
    at.multiselect[0].set_value(["BSM"]).run()
    at.button[0].click().run()
    assert not at.exception, [e.message for e in at.exception]
    assert not at.error, [e.value for e in at.error]
    assert any("Calibration Leaderboard" in s.value for s in at.subheader)
    service = at.session_state["calibration_service"]
    assert service.summary_df.loc["BSM", "Status"] == "Success"
    assert "BSM" in service.model_surfaces
    assert at.dataframe  # leaderboard shown
