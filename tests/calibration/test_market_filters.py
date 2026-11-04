import numpy as np
import pandas as pd
import pytest

from optpricing.atoms import Rate, Stock
from optpricing.calibration import (
    CalibrationSettings,
    LiquidityFilter,
    annotate_chain,
    filter_options,
)
from optpricing.calibration.market_filters import maturity_bucket, moneyness_bucket
from optpricing.models import BSMModel


@pytest.fixture
def chain(synthetic_snapshot):
    return synthetic_snapshot(BSMModel(params={"sigma": 0.25}), r=0.03, q=0.02)


STOCK, RATE = Stock(spot=100.0, dividend=0.02), Rate(rate=0.03)


def test_annotate_chain_adds_forward_moneyness_and_market_iv(chain):
    out = annotate_chain(chain, STOCK, RATE)
    T = out["maturity"].to_numpy()
    np.testing.assert_allclose(out["forward"], 100.0 * np.exp((0.03 - 0.02) * T))
    np.testing.assert_allclose(out["moneyness"], out["strike"] / out["forward"])
    np.testing.assert_allclose(
        out["relative_spread"], (out["ask"] - out["bid"]) / out["marketPrice"]
    )
    # every mid came from sigma = 0.25
    np.testing.assert_allclose(out["iv_market"], 0.25, atol=1e-5)
    assert list(out.index) == list(range(len(out)))


def test_filter_defaults_keep_otm_liquid_options_only(chain):
    kept = filter_options(chain, STOCK, RATE)
    assert not kept.empty
    is_call = kept["optionType"] == "call"
    assert (kept.loc[is_call, "moneyness"] >= 1.0).all()
    assert (kept.loc[~is_call, "moneyness"] < 1.0).all()
    assert kept["maturity"].between(7 / 365.25, 2.0).all()
    assert kept["moneyness"].between(0.7, 1.3).all()


def test_filter_rules_drop_bad_quotes(chain):
    annotated = annotate_chain(chain, STOCK, RATE)
    baseline = filter_options(annotated, STOCK, RATE)
    # a row that survives the default filter, located in the annotated frame
    victim = annotated.index[
        annotated["contractSymbol"] == baseline["contractSymbol"].iloc[0]
    ][0]
    # zero bid
    df = annotated.copy()
    df.loc[victim, "bid"] = 0.0
    assert len(filter_options(df, STOCK, RATE)) == len(baseline) - 1
    # wide spread
    df = annotated.copy()
    df.loc[victim, "relative_spread"] = 0.5
    assert len(filter_options(df, STOCK, RATE)) == len(baseline) - 1
    # no open interest or volume
    df = annotated.copy()
    df.loc[victim, ["openInterest", "volume"]] = 0.0
    assert len(filter_options(df, STOCK, RATE)) == len(baseline) - 1
    # open interest missing but volume present -> kept
    df = annotated.copy()
    df.loc[victim, "openInterest"] = np.nan
    assert len(filter_options(df, STOCK, RATE)) == len(baseline)
    # price below intrinsic has no implied vol -> dropped
    df = annotated.copy()
    df.loc[victim, "iv_market"] = np.nan
    assert len(filter_options(df, STOCK, RATE)) == len(baseline) - 1


def test_filter_windows_are_configurable(chain):
    tight = LiquidityFilter(
        min_maturity_days=60, max_maturity_years=0.6, moneyness=(0.95, 1.05)
    )
    kept = filter_options(chain, STOCK, RATE, tight)
    assert kept["maturity"].between(60 / 365.25, 0.6).all()
    assert kept["moneyness"].between(0.95, 1.05).all()
    both_sides = filter_options(chain, STOCK, RATE, LiquidityFilter(otm_only=False))
    assert len(both_sides) > len(filter_options(chain, STOCK, RATE))


def test_evaluation_filter_is_wider_but_keeps_otm_rule():
    base = LiquidityFilter(moneyness=(0.8, 1.2), max_relative_spread=0.2)
    wide = base.evaluation()
    assert wide.moneyness == pytest.approx((0.7, 1.3))
    assert wide.max_relative_spread == pytest.approx(0.4)
    assert wide.otm_only is True
    assert wide.min_maturity_days == base.min_maturity_days


def test_settings_roundtrip_and_validation():
    s = CalibrationSettings(
        liquidity=LiquidityFilter(moneyness=(0.9, 1.1)), objective="price", n_starts=2
    )
    again = CalibrationSettings.from_dict(s.to_dict())
    assert again == s
    assert s.replace(n_starts=5).n_starts == 5
    assert s.replace(liquidity={"otm_only": False}).liquidity.otm_only is False
    assert s.replace(liquidity={"otm_only": False}).liquidity.moneyness == (0.9, 1.1)
    with pytest.raises(ValueError):
        CalibrationSettings(objective="iv2")
    with pytest.raises(ValueError):
        CalibrationSettings(weighting="x")
    with pytest.raises(ValueError):
        CalibrationSettings(n_starts=0)
    # unknown keys (older cache files) are ignored
    assert (
        CalibrationSettings.from_dict({"objective": "price", "legacy": 1}).objective
        == "price"
    )


def test_buckets():
    assert list(maturity_bucket(pd.Series([0.05, 0.2, 0.4, 0.9, 3.0]))) == [
        "<1m",
        "1-3m",
        "3-6m",
        "6-12m",
        ">12m",
    ]
    assert list(moneyness_bucket(pd.Series([0.7, 0.9, 1.0, 1.1, 1.5]))) == [
        "deep OTM put",
        "OTM put",
        "ATM",
        "OTM call",
        "deep OTM call",
    ]
