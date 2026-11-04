import numpy as np
import pytest

from optpricing.atoms import Rate, Stock
from optpricing.calibration import (
    bucket_metrics,
    evaluate_model,
    filter_options,
    residual_table,
    summarize_residuals,
)
from optpricing.models import BSMModel

STOCK, RATE = Stock(spot=100.0, dividend=0.01), Rate(rate=0.04)


def test_generating_model_has_zero_residuals(bsm_snapshot):
    data = filter_options(bsm_snapshot, STOCK, RATE)
    summary, buckets, table = evaluate_model(
        BSMModel({"sigma": 0.23}), data, STOCK, RATE
    )
    assert summary["n_options"] == len(data)
    assert summary["rmse_price"] < 1e-9
    assert summary["rmse_iv"] < 1e-5
    assert summary["n_iv_failed"] == 0
    assert {
        "model_price",
        "iv_model",
        "price_error",
        "rel_price_error",
        "iv_error",
    } <= set(table.columns)
    assert set(buckets["bucket_type"]) == {"maturity", "moneyness"}
    assert buckets["n_options"].sum() == 2 * len(data)


def test_wrong_vol_shows_in_every_metric(bsm_snapshot):
    data = filter_options(bsm_snapshot, STOCK, RATE)
    summary, buckets, table = evaluate_model(
        BSMModel({"sigma": 0.28}), data, STOCK, RATE
    )
    # flat BSM smile: every option's model IV is 0.28 vs the market's 0.23
    assert summary["rmse_iv"] == pytest.approx(0.05, abs=1e-3)
    assert summary["mae_iv"] == pytest.approx(0.05, abs=1e-3)
    assert summary["rmse_price"] > 0 and summary["rmse_rel_price"] > 0
    assert (buckets["rmse_iv"] > 0.04).all()
    assert (table["price_error"] > 0).all()  # higher vol, higher prices


def test_unpriceable_options_are_counted(bsm_snapshot):
    table = residual_table(BSMModel({"sigma": 0.23}), bsm_snapshot, STOCK, RATE)
    table.loc[table.index[:3], "iv_error"] = np.nan
    summary = summarize_residuals(table)
    assert summary["n_iv_failed"] == 3
    assert np.isfinite(summary["rmse_iv"])


def test_empty_input():
    empty = residual_table(BSMModel(), bsm_empty(), STOCK, RATE)
    assert empty.empty
    assert summarize_residuals(empty)["n_options"] == 0
    assert bucket_metrics(empty).empty


def bsm_empty():
    import pandas as pd

    return pd.DataFrame(columns=["strike", "maturity", "optionType", "marketPrice"])
