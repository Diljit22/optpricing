from unittest.mock import patch

from typer.testing import CliRunner

from optpricing.cli.main import app

runner = CliRunner()


@patch("optpricing.cli.commands.dashboard.subprocess.run")
def test_dashboard_command(mock_subprocess_run):
    """
    Tests that the 'dashboard' command calls streamlit correctly.
    """
    result = runner.invoke(app, ["dashboard"])
    assert result.exit_code == 0
    mock_subprocess_run.assert_called_once()
    assert "streamlit" in mock_subprocess_run.call_args.args[0]
    assert "Home.py" in str(mock_subprocess_run.call_args.args[0][-1])
