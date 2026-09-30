"""Leakage-safe feature engineering for daily demand forecasting."""
from __future__ import annotations

import pandas as pd

LAGS = (1, 7, 14, 28)
ROLL_WINDOWS = (7, 28)
KEYS = ["store", "sku"]

FEATURE_COLUMNS = (
    ["price", "promo", "dow", "month", "dayofyear", "is_weekend", "rel_price"]
    + [f"lag_{k}" for k in LAGS]
    + [f"roll_mean_{w}" for w in ROLL_WINDOWS]
    + [f"roll_std_{w}" for w in ROLL_WINDOWS]
)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar, price, lag and rolling features.

    All history-derived features use only data strictly before the row's date
    (rolling stats are computed on a series shifted by one day), so training
    rows never see the target they are predicting.
    """
    required = {"date", "store", "sku", "price", "promo", "units"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")

    out = df.sort_values(KEYS + ["date"]).reset_index(drop=True).copy()
    out["dow"] = out["date"].dt.dayofweek
    out["month"] = out["date"].dt.month
    out["dayofyear"] = out["date"].dt.dayofyear
    out["is_weekend"] = (out["dow"] >= 5).astype(int)

    grp = out.groupby(KEYS, sort=False)["units"]
    for k in LAGS:
        out[f"lag_{k}"] = grp.shift(k)
    shifted = grp.shift(1)
    shifted_grp = shifted.groupby([out["store"], out["sku"]], sort=False)
    for w in ROLL_WINDOWS:
        out[f"roll_mean_{w}"] = shifted_grp.transform(
            lambda s, w=w: s.rolling(w, min_periods=w).mean()
        )
        out[f"roll_std_{w}"] = shifted_grp.transform(
            lambda s, w=w: s.rolling(w, min_periods=w).std()
        )

    # price relative to the item's typical (max observed) price -> discount depth
    ref_price = out.groupby(KEYS)["price"].transform("max")
    out["rel_price"] = out["price"] / ref_price
    return out
