# Retail Demand Forecasting & Inventory Optimization

![CI](https://github.com/manojdasb/retail-demand-inventory-optimizer/actions/workflows/ci.yml/badge.svg)

An end-to-end, tested pipeline that forecasts daily store x SKU demand and turns
the forecast into an inventory policy (safety stock, reorder point, order-up-to),
then measures the business effect with a service-level simulation.

The problem mirrors a retail supply-chain use case: better forecasts should let
you hold less stock for the same in-stock rate.

> **Data note:** real retailer data is proprietary, so sales are **simulated**
> (`src/rdio/data.py`) with weekly/yearly seasonality, holiday lift, promotions
> with price elasticity and Poisson noise. Results below are on this synthetic
> data; the pipeline is written so a real `date, store, sku, price, promo, units`
> table can replace the generator without other changes.

## Results (seed 42; figures can shift slightly across library versions; 3 stores x 12 SKUs x 730 days = 26,280 rows; 56-day time-based holdout)

| Metric | Naive baseline | Gradient boosting |
|---|---|---|
| WAPE (lower is better) | 22.3% | **14.0%** (37% relative reduction) |
| RMSE (units/day) | 17.8 | **~10.4** |

![Holdout forecast vs actual](reports/forecast_vs_actual.png)

Inventory simulation (95% target service level, 3-day lead time, 36 store-SKU series;
each policy sizes safety stock from its own forecast error):

| | Naive-forecast policy | ML-forecast policy |
|---|---|---|
| Fill rate | 99.95% | ~100% |
| Avg on-hand units | 97.8 | **80.7** (-17%) |

Caveat: fill rates sit above the 95% target because the safety-stock formula is
applied per replenishment cycle and is conservative for Poisson demand. The
relevant comparison is *relative*: same-or-better fill with less inventory.

## Design decisions
- **Time-based split, never random.** Last 56 days are held out.
- **Leakage-safe features.** Lags and rolling stats only use data strictly before
  the target date (unit-tested by perturbing today's value and checking features
  don't change).
- **Poisson-loss gradient boosting** (`HistGradientBoostingRegressor`) because
  demand is non-negative count data.
- **WAPE** as the headline metric: robust to zero-sales days unlike MAPE.
- **Fair baseline.** Naive policy uses a trailing 28-day mean and its own error
  estimate for safety stock.

## Layout
```
src/rdio/data.py       synthetic demand generator
src/rdio/features.py   calendar / lag / rolling features
src/rdio/model.py      time split, metrics, baseline, GBM
src/rdio/inventory.py  safety stock, reorder point, policy simulation
src/rdio/pipeline.py   CLI: end-to-end run -> reports/
tests/                 24 pytest unit tests
.github/workflows/     CI: tests on Python 3.10-3.12 + end-to-end smoke run
```

## Run
```bash
pip install -r requirements.txt && pip install -e .
pytest -q
python -m rdio.pipeline --out reports
```
Outputs `reports/metrics.json`, `reports/policy_report.csv`, `reports/forecast_vs_actual.png`.

## Limitations / next steps
- Single global model across series; hierarchical or per-category models could help.
- Point forecasts only; quantile regression would give prediction intervals directly.
- Lost-sales, no holding/ordering cost optimization (no EOQ) - a cost-based policy is the natural extension.
- Port feature engineering to PySpark/HiveQL for data at scale.
