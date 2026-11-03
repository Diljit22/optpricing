"""Import-time behaviour of ``optpricing.config`` (checked in a fresh interpreter)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

PROBE = textwrap.dedent(
    """
    import json, sys
    import optpricing.config as cfg
    print(json.dumps({
        "root": str(cfg.PROJECT_ROOT),
        "data_dir": str(cfg.DATA_DIR),
        "provider": cfg._config["live_data_provider"],
        "config_exists": cfg.CONFIG_FILE_PATH.exists(),
    }))
    """
)


def _probe(cwd: Path, env_home: str | None) -> dict:
    env = {k: v for k, v in os.environ.items() if k != "OPTPRICING_HOME"}
    if env_home is not None:
        env["OPTPRICING_HOME"] = env_home
    out = subprocess.run(
        [sys.executable, "-c", PROBE],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_import_does_not_write_to_disk(tmp_path):
    """Regression: importing the package used to create config.yaml plus data/
    and artifacts/ next to the installed package (site-packages/lib)."""
    before = sorted(p.name for p in tmp_path.iterdir())
    info = _probe(tmp_path, env_home=None)
    after = sorted(p.name for p in tmp_path.iterdir())

    assert info["root"] == str(tmp_path.resolve())
    assert info["config_exists"] is False
    assert info["provider"] == "yfinance"  # defaults are used in memory
    assert before == after == []


def test_optpricing_home_overrides_cwd(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.yaml").write_text(
        "data_directory: my_data\nlive_data_provider: custom\n"
    )
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()

    info = _probe(elsewhere, env_home=str(home))

    assert info["root"] == str(home.resolve())
    assert info["data_dir"] == str((home / "my_data").resolve())
    assert info["provider"] == "custom"
    assert sorted(p.name for p in home.iterdir()) == [
        "config.yaml"
    ]  # still nothing created
