"""Compare the trained LightGBM models against the L1 imbalance (`obi_0bps`) alone.

For each horizon, on the test rows of the same split as train.py:
    lgbm        the saved `<models>/<label>.txt`, on every factor
    obi_raw     obi_0bps itself, unfitted; only IC and rank IC are meaningful
    obi_linear  fmid = a + b * obi_0bps, least squares on the train rows
    obi_lgbm    LightGBM on obi_0bps alone, with train.py's parameters

Each is scored on every test row and on one row per second (the first in each second).
Run it on the same samples, --max-rows and --split the models were trained with, so the test
rows match.

    uv run baseline.py
    uv run baseline.py --horizons fmid_1s,fmid_5s
"""

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
from scipy.stats import spearmanr

from train import DEFAULT_MODELS, DEFAULT_SAMPLES, load, split

OBI = "obi_0bps"
METHODS = ("lgbm", "obi_raw", "obi_linear", "obi_lgbm")
BP = 1e4


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    p.add_argument("--models", type=Path, default=DEFAULT_MODELS)
    p.add_argument("--horizons", default="all", help="comma-separated fmid columns, or all")
    p.add_argument("--max-rows", type=int, default=None, help="read only the first N rows")
    # the split and LightGBM settings of train.py, so obi_lgbm is trained like the full models
    p.add_argument("--split", default="0.5,0.2,0.3", help="train,valid,test fractions")
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--num-leaves", type=int, default=63)
    p.add_argument("--min-data-in-leaf", type=int, default=1000)
    p.add_argument("--rounds", type=int, default=5000)
    p.add_argument("--early-stopping", type=int, default=100)
    p.add_argument("--threads", type=int, default=0, help="0 lets LightGBM decide")
    return p.parse_args()


def fit(x, y, features, splits, args) -> lgb.Booster:
    """One regressor on the train rows, early-stopped on the valid rows, as in train.py."""
    train_rows, valid_rows, _ = splits
    params = {
        "objective": "regression",
        "learning_rate": args.learning_rate,
        "num_leaves": args.num_leaves,
        "min_data_in_leaf": args.min_data_in_leaf,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "num_threads": args.threads,
        "seed": 0,
        "verbosity": -1,
    }
    train_set = lgb.Dataset(x[train_rows], y[train_rows], feature_name=features, free_raw_data=True)
    valid_set = lgb.Dataset(x[valid_rows], y[valid_rows], reference=train_set)
    return lgb.train(
        params,
        train_set,
        num_boost_round=args.rounds,
        valid_sets=[valid_set],
        valid_names=["valid"],
        callbacks=[lgb.early_stopping(args.early_stopping), lgb.log_evaluation(100)],
    )


def score(y: np.ndarray, pred: np.ndarray, scaled: bool) -> dict:
    """IC and rank IC, plus RMSE in bp and R² when the prediction is a return."""
    resid = y - pred
    return {
        "ic": float(np.corrcoef(y, pred)[0, 1]),
        "rank_ic": float(spearmanr(y, pred).statistic),
        "rmse_bp": float(np.sqrt(resid @ resid / len(y)) * BP) if scaled else float("nan"),
        "r2": float(1.0 - resid @ resid / ((y - y.mean()) @ (y - y.mean()))) if scaled else float("nan"),
    }


def first_per_second(t: np.ndarray) -> np.ndarray:
    _, first = np.unique(t // 1_000_000_000, return_index=True)
    return first


def predictions(label, features, x, y, splits, args) -> dict[str, np.ndarray]:
    """Each method's predictions on the test rows."""
    train_rows, _, test_rows = splits
    j = features.index(OBI)
    obi = np.nan_to_num(x[:, j].astype(np.float64))

    model_path = args.models / f"{label}.txt"
    if not model_path.exists():
        raise SystemExit(f"no model at {model_path}; train it first")
    lgbm = lgb.Booster(model_file=model_path)

    b, a = np.polyfit(obi[train_rows], y[train_rows], 1)
    obi_lgbm = fit(x[:, [j]], y, [OBI], splits, args)

    return {
        "lgbm": lgbm.predict(x[test_rows]),
        "obi_raw": obi[test_rows],
        "obi_linear": a + b * obi[test_rows],
        "obi_lgbm": obi_lgbm.predict(x[test_rows, j : j + 1], num_iteration=obi_lgbm.best_iteration),
    }


def print_table(view: str, metric: str, results: dict) -> None:
    print(f"\n{view}, {metric}\n")
    print("| Horizon | " + " | ".join(METHODS) + " |")
    print("|---|" + "---:|" * len(METHODS))
    for label, by_view in results.items():
        cells = [f"{by_view[view][m][metric]:.4f}" for m in METHODS]
        print(f"| {label.removeprefix('fmid_')} | " + " | ".join(cells) + " |")


def main() -> None:
    args = parse_args()
    labels = None if args.horizons == "all" else [h.strip() for h in args.horizons.split(",") if h.strip()]
    features, labels, x, y, t = load(args.samples, labels, args.max_rows)
    if OBI not in features:
        raise SystemExit(f"{OBI} is not a column of {args.samples}")
    splits = split(t, [float(f) for f in args.split.split(",")])
    test_rows = splits[2]
    per_second = first_per_second(t[test_rows])
    print(f"loaded {len(t)} rows; test has {test_rows.stop - test_rows.start} rows, {len(per_second)} seconds")

    results = {}
    for label in labels:
        preds = predictions(label, features, x, y[label], splits, args)
        y_test = y[label][test_rows]
        results[label] = {
            view: {
                m: score(y_test[rows], preds[m][rows], scaled=m != "obi_raw")
                for m in METHODS
            }
            for view, rows in (("all rows", slice(None)), ("1 per second", per_second))
        }

    for view in ("all rows", "1 per second"):
        for metric in ("ic", "rank_ic", "rmse_bp"):
            print_table(view, metric, results)
    out = args.models / "baseline.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
