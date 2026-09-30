import numpy as np
import pytest
from rdio.inventory import (reorder_point, safety_stock, simulate_policy,
                            z_for_service)


def test_z_score_95():
    assert z_for_service(0.95) == pytest.approx(1.6449, abs=1e-3)


def test_safety_stock_formula():
    assert safety_stock(10, 4, 0.95) == pytest.approx(1.6449 * 10 * 2, abs=0.05)


def test_safety_stock_rises_with_service_level():
    assert safety_stock(5, 3, 0.99) > safety_stock(5, 3, 0.90)


def test_reorder_point_adds_lead_time_demand():
    assert reorder_point(20, 0, 5, 0.95) == pytest.approx(100)


def test_invalid_inputs():
    with pytest.raises(ValueError):
        z_for_service(1.0)
    with pytest.raises(ValueError):
        safety_stock(-1, 3, 0.9)


def test_perfect_forecast_constant_demand_full_fill():
    d = np.full(60, 10.0)
    r = simulate_policy(d, d, sigma=0.0, lead_time=3, service_level=0.95)
    assert r["fill_rate"] == pytest.approx(1.0)


def test_higher_service_level_gives_higher_fill_and_inventory():
    rng = np.random.default_rng(0)
    actual = rng.poisson(20, 400).astype(float)
    fc = np.full(400, 20.0)
    lo = simulate_policy(actual, fc, sigma=4.5, service_level=0.80)
    hi = simulate_policy(actual, fc, sigma=4.5, service_level=0.99)
    assert hi["fill_rate"] >= lo["fill_rate"]
    assert hi["avg_on_hand"] > lo["avg_on_hand"]


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        simulate_policy(np.ones(5), np.ones(4), 1.0)


def test_policy_report_uses_each_policys_own_sigma():
    import pandas as pd
    from rdio.inventory import policy_report
    n = 60
    t = pd.DataFrame({"store": "S00", "sku": "A",
                      "date": pd.date_range("2024-01-01", periods=n),
                      "units": np.full(n, 10.0), "roll_mean_28": np.full(n, 10.0)})
    key = pd.MultiIndex.from_tuples([("S00", "A")])
    small = pd.Series([1.0], index=key)
    big = pd.Series([6.0], index=key)
    rep = policy_report(t, np.full(n, 10.0), small, big)
    assert rep.loc[0, "naive_inv"] > rep.loc[0, "ml_inv"]
