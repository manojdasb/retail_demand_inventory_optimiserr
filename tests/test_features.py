import numpy as np
import pandas as pd
import pytest
from rdio.data import generate_sales
from rdio.features import FEATURE_COLUMNS, build_features


def _feats():
    return build_features(generate_sales(n_skus=2, n_stores=2, days=120))


def test_lag_matches_manual_shift():
    f = _feats()
    g = f[(f.store == "S00") & (f.sku == "SKU000")].reset_index(drop=True)
    assert g.loc[10, "lag_7"] == g.loc[3, "units"]


def test_lags_do_not_leak_across_series():
    f = _feats()
    first_rows = f.groupby(["store", "sku"]).head(7)
    assert first_rows["lag_7"].isna().all()


def test_rolling_mean_excludes_current_day():
    f = _feats()
    g = f[(f.store == "S00") & (f.sku == "SKU000")].reset_index(drop=True)
    expected = g.loc[20 - 7 : 20 - 1, "units"].mean()
    assert np.isclose(g.loc[20, "roll_mean_7"], expected)


def test_changing_today_does_not_change_today_features():
    df = generate_sales(n_skus=1, n_stores=1, days=100)
    a = build_features(df)
    df2 = df.copy()
    df2.loc[60, "units"] += 1000
    b = build_features(df2)
    for c in FEATURE_COLUMNS:
        if c.startswith(("lag_", "roll_")):
            assert (a.loc[60, c] == b.loc[60, c]) or (pd.isna(a.loc[60, c]) and pd.isna(b.loc[60, c]))


def test_missing_columns_raises():
    with pytest.raises(ValueError):
        build_features(pd.DataFrame({"date": []}))
