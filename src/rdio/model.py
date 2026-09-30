"""Forecasting models, time-based validation and metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .features import FEATURE_COLUMNS


def time_split(df: pd.DataFrame, holdout_days: int = 56):
    """Split by time (never randomly): the last `holdout_days` are the test set."""
    if holdout_days < 1:
        raise ValueError("holdout_days must be >= 1")
    cutoff = df["date"].max() - pd.Timedelta(days=holdout_days)
    train = df[df["date"] <= cutoff].dropna(subset=FEATURE_COLUMNS)
    test = df[df["date"] > cutoff].dropna(subset=FEATURE_COLUMNS)
    if train.empty or test.empty:
        raise ValueError("split produced an empty train or test set")
    return train, test


def wape(y_true, y_pred) -> float:
    """Weighted absolute percentage error: sum|e| / sum|y|. Robust to zero sales."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.abs(y_true).sum()
    if denom == 0:
        raise ValueError("WAPE undefined when all actuals are zero")
    return float(np.abs(y_true - y_pred).sum() / denom)


def rmse(y_true, y_pred) -> float:
    d = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean(d**2)))


def seasonal_naive(df: pd.DataFrame) -> np.ndarray:
    """Baseline: predict the same weekday last week."""
    return df["lag_7"].to_numpy()


def fit_gbm(train: pd.DataFrame, seed: int = 0) -> HistGradientBoostingRegressor:
    model = HistGradientBoostingRegressor(
        loss="poisson",
        max_iter=300,
        learning_rate=0.06,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=seed,
    )
    model.fit(train[FEATURE_COLUMNS], train["units"])
    return model


def evaluate(train: pd.DataFrame, test: pd.DataFrame, seed: int = 0) -> dict:
    """Fit the GBM, compare against the seasonal-naive baseline on the holdout."""
    model = fit_gbm(train, seed=seed)
    pred = model.predict(test[FEATURE_COLUMNS])
    base = seasonal_naive(test)
    metrics = {
        "gbm_wape": wape(test["units"], pred),
        "gbm_rmse": rmse(test["units"], pred),
        "naive_wape": wape(test["units"], base),
        "naive_rmse": rmse(test["units"], base),
    }
    metrics["wape_improvement_pct"] = 100 * (
        1 - metrics["gbm_wape"] / metrics["naive_wape"]
    )
    return {"model": model, "pred": pred, "metrics": metrics}
