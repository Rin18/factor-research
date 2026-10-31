"""Train one LightGBM regressor per fmid horizon on the factor samples.

Rows are split by time into train / valid / test (50/20/30 by default), with a purge gap at
each boundary so no label reaches into the next split. The label is the raw fmid return.
Each model is saved in LightGBM's text format as `<models>/<label>.txt`, next to its metrics.

    uv run train.py                                         # every fmid horizon
    uv run train.py --horizons fmid_1s,fmid_5s
    uv run train.py --horizons fmid_1s --max-rows 1000000   # quick workflow check
"""

import argparse
import json
import math
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pyarrow.parquet as pq
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parent.parent
DEFAULT_SAMPLES = REPO / "samples" / "samples.parquet"
DEFAULT_MODELS = Path(__file__).resolve().parent / "models"

# columns that are neither labels nor factors
NON_FEATURES = {"time", "time_ns", "event"}
LABEL_PREFIX = "fmid_"
# no label looks further ahead than the longest horizon
PURGE_NS = 60 * 1_000_000_000


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    p.add_argument("--models", type=Path, default=DEFAULT_MODELS)
    p.add_argument("--horizons", default="all", help="comma-separated fmid columns, or all")
    p.add_argument("--max-rows", type=int, default=None, help="read only the first N rows")
    p.add_argument("--split", default="0.5,0.2,0.3", help="train,valid,test fractions")
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--num-leaves", type=int, default=63)
    p.add_argument("--min-data-in-leaf", type=int, default=1000)
    p.add_argument("--rounds", type=int, default=5000)
    p.add_argument("--early-stopping", type=int, default=100)
    p.add_argument("--threads", type=int, default=0, help="0 lets LightGBM decide")
    return p.parse_args()


def load(path: Path, labels: list[str] | None, max_rows: int | None):
    """Factors as float32 in file column order, the requested labels (every fmid column if
    None), and time_ns."""
    file = pq.ParquetFile(path)
    names = file.schema_arrow.names
    features = [c for c in names if c not in NON_FEATURES and not c.startswith(LABEL_PREFIX)]
    if labels is None:
        labels = [c for c in names if c.startswith(LABEL_PREFIX)]
    missing = [label for label in labels if label not in names]
    if missing:
        raise SystemExit(f"labels not in {path}: {missing}")

    n = file.metadata.num_rows if max_rows is None else min(max_rows, file.metadata.num_rows)
    x = np.empty((n, len(features)), dtype=np.float32)
    y = {label: np.empty(n, dtype=np.float64) for label in labels}
    t = np.empty(n, dtype=np.int64)
    row = 0
    # one row group at a time; iter_batches buffers far ahead and blows up memory
    for group in range(file.metadata.num_row_groups):
        if row == n:
            break
        batch = file.read_row_group(group, columns=["time_ns", *features, *labels])
        take = min(batch.num_rows, n - row)
        batch = batch.slice(0, take)
        t[row : row + take] = batch.column("time_ns").to_numpy()
        for j, name in enumerate(features):
            x[row : row + take, j] = batch.column(name).to_numpy(zero_copy_only=False)
        for label in labels:
            y[label][row : row + take] = batch.column(label).to_numpy(zero_copy_only=False)
        row += take
    if np.any(np.diff(t) < 0):
        raise SystemExit("samples are not sorted by time_ns")
    return features, labels, x, y, t


def split(t: np.ndarray, fractions: list[float]):
    """Row ranges of train / valid / test, cut by row count, minus a purge gap before each cut."""
    if len(fractions) != 3 or not math.isclose(sum(fractions), 1.0):
        raise SystemExit(f"split must be three fractions summing to 1, got {fractions}")
    n = len(t)
    cut1 = int(n * fractions[0])
    cut2 = int(n * (fractions[0] + fractions[1]))
    end1 = int(np.searchsorted(t, t[cut1] - PURGE_NS))
    end2 = int(np.searchsorted(t, t[cut2] - PURGE_NS))
    return slice(0, end1), slice(cut1, end2), slice(cut2, n)


def metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    resid = y - pred
    return {
        "rows": int(len(y)),
        "r2": float(1.0 - resid @ resid / ((y - y.mean()) @ (y - y.mean()))),
        "ic": float(np.corrcoef(y, pred)[0, 1]),
        "rank_ic": float(spearmanr(y, pred).statistic),
    }


def train(label, features, x, y, splits, args) -> None:
    train_rows, valid_rows, test_rows = splits
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
    booster = lgb.train(
        params,
        train_set,
        num_boost_round=args.rounds,
        valid_sets=[valid_set],
        valid_names=["valid"],
        callbacks=[lgb.early_stopping(args.early_stopping), lgb.log_evaluation(100)],
    )

    result = {"label": label, "best_iteration": booster.best_iteration, "unit": "return"}
    for name, rows in (("train", train_rows), ("valid", valid_rows), ("test", test_rows)):
        result[name] = metrics(y[rows], booster.predict(x[rows], num_iteration=booster.best_iteration))
    gain = booster.feature_importance("gain", iteration=booster.best_iteration)
    result["importance_gain"] = dict(sorted(zip(features, gain.tolist()), key=lambda kv: -kv[1]))

    args.models.mkdir(parents=True, exist_ok=True)
    model_path = args.models / f"{label}.txt"
    booster.save_model(model_path, num_iteration=booster.best_iteration)
    (args.models / f"{label}.metrics.json").write_text(json.dumps(result, indent=2))

    print(f"\n{label}: best_iteration={booster.best_iteration}, saved {model_path}")
    for name in ("train", "valid", "test"):
        m = result[name]
        print(f"  {name:5} rows={m['rows']:>9}  r2={m['r2']:+.5f}  ic={m['ic']:+.4f}  rank_ic={m['rank_ic']:+.4f}")
    print("  top features by gain:")
    for name, g in list(result["importance_gain"].items())[:15]:
        print(f"    {name:32} {g:.4g}")


def main() -> None:
    args = parse_args()
    labels = None if args.horizons == "all" else [h.strip() for h in args.horizons.split(",") if h.strip()]
    features, labels, x, y, t = load(args.samples, labels, args.max_rows)
    splits = split(t, [float(f) for f in args.split.split(",")])
    print(f"loaded {len(t)} rows x {len(features)} factors from {args.samples}")
    for name, rows in zip(("train", "valid", "test"), splits):
        print(f"  {name:5} rows {rows.start}..{rows.stop}")
    for label in labels:
        train(label, features, x, y[label], splits, args)


if __name__ == "__main__":
    main()
