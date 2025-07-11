from unittest.mock import MagicMock, call, patch

import pytest

from optpricing.dashboard.widgets import build_sidebar


@pytest.fixture
def mock_streamlit():
    """Mock Streamlit functions and query_params."""
    with (
        patch("streamlit.sidebar") as mock_sidebar,
        patch("streamlit.query_params", new=MagicMock()) as mock_query_params,
        patch("streamlit.selectbox") as mock_selectbox,
        patch("streamlit.header") as mock_header,
        patch("streamlit.markdown") as mock_markdown,
    ):
        mock_sidebar_context = MagicMock()
        mock_sidebar.__enter__.return_value = mock_sidebar_context
        mock_sidebar.__exit__.return_value = None
        yield {
            "sidebar": mock_sidebar,
            "query_params": mock_query_params,
            "selectbox": mock_selectbox,
            "header": mock_header,
            "markdown": mock_markdown,
            "sidebar_context": mock_sidebar_context,
        }


@pytest.fixture
def mock_config_and_data():
    """Mock config and get_available_snapshot_dates."""
    with (
        patch("optpricing.dashboard.widgets._config") as mock_config,
        patch(
            "optpricing.dashboard.widgets.get_available_snapshot_dates"
        ) as mock_get_dates,
        patch(
            "optpricing.dashboard.widgets.ALL_MODEL_CONFIGS",
            {"BSM": {}, "Merton": {}},
        ),
    ):
        mock_config.get.side_effect = [
            ["SPY", "AAPL", "TSLA", "NVDA"],
            ["SPY", "AAPL", "TSLA", "NVDA"],
        ]
        mock_get_dates.return_value = ["2023-01-01", "2023-02-01"]
        yield mock_config, mock_get_dates


def test_build_sidebar_default_selections(mock_streamlit, mock_config_and_data):
    """
    Test build_sidebar with default selections and no query params.
    """
    mock_config, mock_get_dates = mock_config_and_data
    mock_streamlit["query_params"].get.side_effect = ["SPY", "Live Data", "BSM"]
    mock_streamlit["selectbox"].side_effect = ["SPY", "Live Data", "BSM"]

    result = build_sidebar()

    assert result == ("SPY", "Live Data", "BSM")
    mock_streamlit["header"].assert_called_once_with("Global Configuration")
    assert mock_streamlit["selectbox"].call_count == 3
    mock_streamlit["selectbox"].assert_any_call(
        "Ticker", ["SPY", "AAPL", "TSLA", "NVDA"], index=0, key="ticker_selector"
    )
    mock_streamlit["selectbox"].assert_any_call(
        "Snapshot Date",
        ["Live Data", "2023-01-01", "2023-02-01"],
        index=0,
        key="date_selector",
    )
    mock_streamlit["selectbox"].assert_any_call(
        "Model", ["BSM", "Merton"], index=0, key="model_selector"
    )
    assert mock_streamlit["markdown"].call_count == 5
    mock_streamlit["query_params"].__setitem__.assert_has_calls(
        [
            call("ticker", "SPY"),
            call("date", "Live Data"),
            call("model", "BSM"),
        ]
    )


def test_build_sidebar_custom_query_params(mock_streamlit, mock_config_and_data):
    """
    Test build_sidebar with custom query params.
    """
    mock_config, mock_get_dates = mock_config_and_data
    mock_streamlit["query_params"].get.side_effect = ["AAPL", "2023-01-01", "Merton"]
    mock_streamlit["selectbox"].side_effect = ["AAPL", "2023-01-01", "Merton"]

    result = build_sidebar()

    assert result == ("AAPL", "2023-01-01", "Merton")
    mock_streamlit["selectbox"].assert_any_call(
        "Ticker", ["SPY", "AAPL", "TSLA", "NVDA"], index=1, key="ticker_selector"
    )
    mock_streamlit["selectbox"].assert_any_call(
        "Snapshot Date",
        ["Live Data", "2023-01-01", "2023-02-01"],
        index=1,
        key="date_selector",
    )
    mock_streamlit["selectbox"].assert_any_call(
        "Model", ["BSM", "Merton"], index=1, key="model_selector"
    )
    mock_streamlit["query_params"].__setitem__.assert_has_calls(
        [
            call("ticker", "AAPL"),
            call("date", "2023-01-01"),
            call("model", "Merton"),
        ]
    )


def test_build_sidebar_invalid_ticker(mock_streamlit, mock_config_and_data):
    """
    Test build_sidebar with invalid ticker in query params.
    """
    mock_config, mock_get_dates = mock_config_and_data
    mock_streamlit["query_params"].get.side_effect = ["INVALID", "Live Data", "BSM"]
    mock_streamlit["selectbox"].side_effect = ["SPY", "Live Data", "BSM"]

    result = build_sidebar()

    assert result == ("SPY", "Live Data", "BSM")
    mock_streamlit["selectbox"].assert_any_call(
        "Ticker", ["SPY", "AAPL", "TSLA", "NVDA"], index=0, key="ticker_selector"
    )


def test_build_sidebar_data_source_error(mock_streamlit, mock_config_and_data):
    """
    Test build_sidebar when get_available_snapshot_dates raises FileNotFoundError.
    """
    mock_config, mock_get_dates = mock_config_and_data
    mock_get_dates.side_effect = FileNotFoundError("No data")
    mock_streamlit["query_params"].get.side_effect = ["SPY", "Live Data", "BSM"]
    mock_streamlit["selectbox"].side_effect = ["SPY", "Live Data", "BSM"]

    result = build_sidebar()

    assert result == ("SPY", "Live Data", "BSM")
    mock_streamlit["selectbox"].assert_any_call(
        "Snapshot Date", ["Live Data"], index=0, key="date_selector"
    )


def test_build_sidebar_invalid_date_and_model(mock_streamlit, mock_config_and_data):
    """
    Test build_sidebar with invalid date and model in query params.
    """
    mock_config, mock_get_dates = mock_config_and_data
    mock_streamlit["query_params"].get.side_effect = [
        "SPY",
        "INVALID_DATE",
        "INVALID_MODEL",
    ]
    mock_streamlit["selectbox"].side_effect = [
        "SPY",
        "Live Data",
        "BSM",
    ]

    result = build_sidebar()

    assert result == ("SPY", "Live Data", "BSM")
    mock_streamlit["selectbox"].assert_any_call(
        "Snapshot Date",
        ["Live Data", "2023-01-01", "2023-02-01"],
        index=0,
        key="date_selector",
    )
    mock_streamlit["selectbox"].assert_any_call(
        "Model", ["BSM", "Merton"], index=0, key="model_selector"
    )
