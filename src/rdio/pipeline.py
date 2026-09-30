"""End-to-end run: generate data -> features -> forecast -> inventory simulation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .data import generate_sales  # noqa: E402
from .features import FEATURE_COLUMNS, build_features  # noqa: E402
from .inventory import policy_report  # noqa: E402
from .model import evaluate, fit_gbm, time_split  # noqa: E402


def run(out_dir: str = "reports", n_skus: int = 12, n_stores: int = 3,
        days: int = 730, seed: int = 42, lead_time: int = 3,
        service_level: float = 0.95) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    sales = generate_sales(n_skus=n_skus, n_stores=n_stores, days=days, seed=seed)
    feats = build_features(sales)
    train, test = time_split(feats, holdout_days=56)

    res = evaluate(train, test, seed=seed)
    metrics = res["metrics"]

    # per-series residual sigma from the in-sample fit (used for safety stock)
    tr_model = fit_gbm(train, seed=seed)
    resid = train["units"] - tr_model.predict(train[FEATURE_COLUMNS])
    sigma_ml = resid.groupby([train["store"], train["sku"]]).std()
    resid_naive = train["units"] - train["roll_mean_28"]
    sigma_naive = resid_naive.groupby([train["store"], train["sku"]]).std()

    rep = policy_report(test, res["pred"], sigma_ml, sigma_naive,
                        lead_time, service_level)
    metrics.update({
        "n_rows": int(len(sales)),
        "n_series": int(rep.shape[0]),
        "mean_fill_ml": float(rep["ml_fill"].mean()),
        "mean_fill_naive": float(rep["naive_fill"].mean()),
        "mean_inv_ml": float(rep["ml_inv"].mean()),
        "mean_inv_naive": float(rep["naive_inv"].mean()),
        "inventory_change_pct": float(
            100 * (rep["ml_inv"].mean() / rep["naive_inv"].mean() - 1)),
        "target_service_level": service_level,
        "lead_time_days": lead_time,
    })

    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    rep.to_csv(out / "policy_report.csv", index=False)

    # plot: actual vs forecast for the highest-volume series
    top = test.groupby(["store", "sku"])["units"].sum().idxmax()
    mask = (test["store"] == top[0]) & (test["sku"] == top[1])
    g = test[mask].assign(pred=res["pred"][mask.to_numpy()]).sort_values("date")
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(g["date"], g["units"], label="Actual", lw=1.4)
    ax.plot(g["date"], g["pred"], label="GBM forecast", lw=1.4)
    ax.set_title(f"Holdout forecast - {top[0]} / {top[1]}")
    ax.set_ylabel("Units/day")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out / "forecast_vs_actual.png", dpi=130)
    plt.close(fig)
    return metrics


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="reports")
    p.add_argument("--n-skus", type=int, default=12)
    p.add_argument("--n-stores", type=int, default=3)
    p.add_argument("--days", type=int, default=730)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lead-time", type=int, default=3)
    p.add_argument("--service-level", type=float, default=0.95)
    a = p.parse_args()
    m = run(a.out, a.n_skus, a.n_stores, a.days, a.seed, a.lead_time, a.service_level)
    print(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
