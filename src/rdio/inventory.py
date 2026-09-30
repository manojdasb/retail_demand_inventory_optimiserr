"""Inventory policy: safety stock, reorder points and a service-level simulation."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


def z_for_service(service_level: float) -> float:
    if not 0.5 < service_level < 1.0:
        raise ValueError("service_level must be in (0.5, 1.0)")
    return float(norm.ppf(service_level))


def safety_stock(sigma_daily: float, lead_time_days: int, service_level: float) -> float:
    """z * sigma_daily * sqrt(L): standard safety stock under iid daily forecast error."""
    if sigma_daily < 0 or lead_time_days < 1:
        raise ValueError("sigma_daily>=0 and lead_time_days>=1 required")
    return z_for_service(service_level) * sigma_daily * np.sqrt(lead_time_days)


def reorder_point(mean_daily: float, sigma_daily: float, lead_time_days: int,
                  service_level: float) -> float:
    return mean_daily * lead_time_days + safety_stock(
        sigma_daily, lead_time_days, service_level
    )


def simulate_policy(
    actual: np.ndarray,
    forecast: np.ndarray,
    sigma: float,
    lead_time: int = 3,
    service_level: float = 0.95,
    review_period: int = 1,
) -> dict:
    """Order-up-to policy simulation over a demand series.

    Every `review_period` days, order enough to raise inventory position to
    forecast demand over (lead_time + review_period) days plus safety stock.
    Orders arrive after `lead_time` days. Lost sales are not backordered.

    Returns fill rate (units served / units demanded), stockout-day rate and
    average on-hand inventory.
    """
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    if len(actual) != len(forecast):
        raise ValueError("actual and forecast must be the same length")
    n = len(actual)
    horizon = lead_time + review_period
    ss = safety_stock(sigma, horizon, service_level)
    # extend the forecast past the end of the series so look-ahead windows are
    # always full (repeats the last forecast value)
    forecast = np.concatenate([forecast, np.full(horizon + 1, forecast[-1])])

    on_hand = float(forecast[:horizon].sum() + ss)
    pipeline = np.zeros(n + lead_time + 1)
    served = demanded = 0.0
    stockout_days = 0
    inv_sum = 0.0
    for t in range(n):
        on_hand += pipeline[t]
        d = actual[t]
        sold = min(on_hand, d)
        served += sold
        demanded += d
        stockout_days += int(sold < d)
        on_hand -= sold
        inv_sum += on_hand
        if t % review_period == 0:
            window = forecast[t + 1 : t + 1 + horizon]
            target = window.sum() + ss
            position = on_hand + pipeline[t + 1 : t + lead_time + 1].sum()
            order = max(0.0, target - position)
            pipeline[t + lead_time] += order
    return {
        "fill_rate": served / demanded if demanded else 1.0,
        "stockout_day_rate": stockout_days / n,
        "avg_on_hand": inv_sum / n,
    }


def policy_report(test: pd.DataFrame, pred: np.ndarray,
                  sigma_ml: pd.Series, sigma_naive: pd.Series,
                  lead_time: int = 3, service_level: float = 0.95) -> pd.DataFrame:
    """Per store-SKU simulation: ML-forecast policy vs. a naive-forecast policy.

    Naive = trailing 28-day mean as the forecast. Each policy sizes its safety
    stock from *its own* historical forecast error, so the comparison is fair:
    a more accurate forecast should need less buffer for the same service.
    """
    df = test.copy()
    df["pred"] = pred
    rows = []
    for (store, sku), g in df.groupby(["store", "sku"]):
        g = g.sort_values("date")
        actual = g["units"].to_numpy()
        ml = simulate_policy(actual, g["pred"].to_numpy(),
                             float(sigma_ml[(store, sku)]), lead_time, service_level)
        nv = simulate_policy(actual, g["roll_mean_28"].to_numpy(),
                             float(sigma_naive[(store, sku)]), lead_time, service_level)
        rows.append({"store": store, "sku": sku,
                     "ml_fill": ml["fill_rate"], "naive_fill": nv["fill_rate"],
                     "ml_inv": ml["avg_on_hand"], "naive_inv": nv["avg_on_hand"]})
    return pd.DataFrame(rows)
