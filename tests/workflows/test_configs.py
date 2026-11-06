import pytest

from optpricing.models import BatesModel
from optpricing.workflows.configs import (
    ALL_MODEL_CONFIGS,
    calibratable_model_names,
    get_model_config,
    resolve_model_name,
)


@pytest.mark.parametrize("name", list(ALL_MODEL_CONFIGS))
def test_recipe_is_well_formed(name):
    cfg = ALL_MODEL_CONFIGS[name]
    assert cfg["name"] == name
    model_class = cfg["model_class"]
    guess = dict(cfg.get("initial_guess") or {})
    bounds = dict(cfg.get("bounds") or {})
    frozen = cfg.get("frozen", {})
    assert isinstance(frozen, dict), "frozen must map names to values"
    # the initial guess builds a valid model and sits inside the bounds
    params = {**{k: v for k, v in model_class.default_params.items()}, **guess}
    model_class(params=params)
    for key, value in guess.items():
        if key in bounds:
            lo, hi = bounds[key]
            assert lo <= value <= hi, (
                f"{name}.{key} guess {value} outside {bounds[key]}"
            )
    for key in bounds:
        assert key in model_class.default_params or key in guess


@pytest.mark.parametrize("name", calibratable_model_names())
def test_every_fitted_parameter_moves_prices(name):
    """A fitted parameter the prices do not depend on is unidentifiable and
    lets the optimiser wander (regression: the hyperbolic recipe fitted the
    location mu, which the risk-neutral drift correction cancels exactly)."""
    import pandas as pd

    from optpricing import Rate, Stock
    from optpricing.calibration.vectorized_pricer import price_options_vectorized

    cfg = ALL_MODEL_CONFIGS[name]
    model_class = cfg["model_class"]
    base = {**model_class.default_params, **(cfg.get("initial_guess") or {})}
    base.update(cfg.get("frozen", {}))
    chain = pd.DataFrame(
        {
            "strike": [80.0, 100.0, 120.0] * 2,
            "maturity": [0.25] * 3 + [1.0] * 3,
            "optionType": ["put", "call", "call"] * 2,
        }
    )
    stock, rate = Stock(spot=100.0, dividend=0.01), Rate(rate=0.03)

    def prices(params):
        return price_options_vectorized(
            chain, stock, model_class(params=params), rate, **params
        )

    reference = prices(base)
    for key, (lo, hi) in cfg["bounds"].items():
        if key in cfg.get("frozen", {}):
            continue
        moved = []
        for target in (base[key] + 0.1 * (hi - lo), base[key] - 0.1 * (hi - lo)):
            try:
                moved.append(prices({**base, key: min(max(target, lo), hi)}))
            except ValueError:  # e.g. |beta| < alpha
                continue
        assert moved, f"{name}.{key}: no valid perturbation"
        change = max(abs(m - reference).max() for m in moved)
        assert change > 1e-6, f"{name}.{key} does not affect prices"


def test_bates_param_defs_do_not_include_sigma():
    assert "sigma" not in BatesModel.param_defs
    assert set(BatesModel.param_defs) == set(BatesModel.default_params)


def test_vg_key_and_aliases():
    assert "VG" in ALL_MODEL_CONFIGS and "Variance Gamma" not in ALL_MODEL_CONFIGS
    assert resolve_model_name("Variance Gamma") == "VG"
    assert resolve_model_name("vg") == "VG"
    assert resolve_model_name("heston") == "Heston"
    assert resolve_model_name("Black-Scholes-Merton") == "BSM"
    assert resolve_model_name("SABR with Jumps") == "SABRJump"
    with pytest.raises(KeyError, match="Unknown model"):
        resolve_model_name("Nope")
    cfg = get_model_config("merton")
    cfg["ticker"] = "X"  # a copy: the shared recipe is untouched
    assert "ticker" not in ALL_MODEL_CONFIGS["Merton"]


def test_calibratable_sets():
    without_mc = calibratable_model_names()
    assert set(without_mc) == {
        "BSM",
        "Merton",
        "Heston",
        "Bates",
        "SABR",
        "CEV",
        "Kou",
        "NIG",
        "VG",
        "CGMY",
        "Hyperbolic",
    }
    assert set(calibratable_model_names(allow_mc=True)) == set(ALL_MODEL_CONFIGS)
    assert ALL_MODEL_CONFIGS["SABRJump"]["requires_mc"] is True


def test_heston_start_point_satisfies_feller():
    g = ALL_MODEL_CONFIGS["Heston"]["initial_guess"]
    assert 2 * g["kappa"] * g["theta"] >= g["vol_of_vol"] ** 2
