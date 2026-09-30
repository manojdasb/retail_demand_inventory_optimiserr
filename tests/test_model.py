import numpy as np
import pytest
from rdio.data import generate_sales
from rdio.features import build_features
from rdio.model import evaluate, rmse, time_split, wape


def test_wape_known_value():
    assert wape([10, 10], [8, 12]) == pytest.approx(0.2)


def test_wape_zero_actuals_raises():
    with pytest.raises(ValueError):
        wape([0, 0], [1, 1])


def test_rmse_known_value():
    assert rmse([0, 0], [3, 4]) == pytest.approx(np.sqrt(12.5))


def test_time_split_has_no_overlap():
    f = build_features(generate_sales(n_skus=2, n_stores=1, days=200))
    train, test = time_split(f, holdout_days=30)
    assert train["date"].max() < test["date"].min()


def test_gbm_beats_seasonal_naive():
    f = build_features(generate_sales(n_skus=4, n_stores=2, days=500))
    train, test = time_split(f, holdout_days=56)
    m = evaluate(train, test)["metrics"]
    assert m["gbm_wape"] < m["naive_wape"]
