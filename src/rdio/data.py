"""Synthetic retail sales generator.

Real retailer data is proprietary, so this module simulates daily store x SKU
demand with the structure real retail data has: weekly and yearly seasonality,
holiday lift, promotions with price elasticity, per-SKU base rates and noise.
Demand is drawn from a Poisson distribution so it is count-valued and
heteroscedastic, like unit sales.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HOLIDAYS = [(1, 1), (7, 4), (11, 25), (12, 24), (12, 25)]  # (month, day)


def generate_sales(
    n_skus: int = 12,
    n_stores: int = 3,
    days: int = 730,
    start: str = "2023-01-01",
    seed: int = 42,
) -> pd.DataFrame:
    """Return long-format daily sales: date, store, sku, price, promo, units."""
    if n_skus < 1 or n_stores < 1 or days < 60:
        raise ValueError("need n_skus>=1, n_stores>=1, days>=60")
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=days, freq="D")
    frames = []
    for store in range(n_stores):
        store_scale = rng.uniform(0.7, 1.5)
        for sku in range(n_skus):
            base = rng.uniform(8, 60) * store_scale
            list_price = rng.uniform(3, 40)
            elasticity = rng.uniform(1.2, 2.5)
            dow_amp = rng.uniform(0.10, 0.35)
            year_amp = rng.uniform(0.05, 0.30)
            phase = rng.uniform(0, 2 * np.pi)

            t = np.arange(days)
            weekly = 1 + dow_amp * np.sin(2 * np.pi * (dates.dayofweek.values - 4) / 7)
            yearly = 1 + year_amp * np.sin(2 * np.pi * t / 365.25 + phase)
            is_holiday = np.array([(d.month, d.day) in HOLIDAYS for d in dates])
            holiday = 1 + 0.35 * is_holiday

            promo = (rng.random(days) < 0.10).astype(int)
            discount = np.where(promo == 1, rng.uniform(0.10, 0.30, days), 0.0)
            price = list_price * (1 - discount)
            price_effect = (price / list_price) ** (-elasticity)

            lam = base * weekly * yearly * holiday * price_effect
            units = rng.poisson(lam)
            frames.append(
                pd.DataFrame(
                    {
                        "date": dates,
                        "store": f"S{store:02d}",
                        "sku": f"SKU{sku:03d}",
                        "price": price.round(2),
                        "promo": promo,
                        "units": units,
                    }
                )
            )
    return pd.concat(frames, ignore_index=True)
