"""The README and the validation guide quote generated files, not hand-typed numbers."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _span(text: str, name: str) -> str:
    start, end = f"<!-- {name} -->", f"<!-- /{name} -->"
    i, j = text.find(start), text.find(end)
    assert i >= 0 and j > i, name
    return text[i + len(start) : j].strip()


def test_readme_accuracy_matches_json():
    accuracy = _load("accuracy_tables", ROOT / "benchmarks" / "accuracy.py")
    payload = json.loads((ROOT / "benchmarks" / "accuracy.json").read_text())
    readme = (ROOT / "README.md").read_text()
    guide = (ROOT / "docs" / "guide" / "validation.md").read_text()
    assert _span(readme, "accuracy-summary") == accuracy.render_summary(
        payload["rows"], worst=False
    )
    assert _span(guide, "accuracy-summary-full") == accuracy.render_summary(
        payload["rows"], worst=True
    )
    slopes = accuracy.render_convergence(payload["convergence"])
    assert _span(readme, "accuracy-convergence") == slopes
    assert _span(guide, "accuracy-convergence") == slopes


def test_readme_performance_matches_json():
    bench = _load("bench_tables", ROOT / "benchmarks" / "run_benchmarks.py")
    payload = json.loads(
        (ROOT / "benchmarks" / "results" / "2026-11-18-arm64.json").read_text()
    )
    readme = (ROOT / "README.md").read_text()
    assert _span(readme, "perf-table") == bench.markdown_table(payload)
