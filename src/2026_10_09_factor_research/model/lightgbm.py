"""Combine all factors with LightGBM; write one test Rank IC heatmap PNG.

Run: uv run python src/2026_10_09_factor_research/model/lightgbm.py
Read the symbol and split from the adjacent config.toml. Training parameters
are defined in NUM_BOOST_ROUND and LIGHTGBM_PARAMS below. Fit one model per horizon.
"""

from fractions import Fraction
from pathlib import Path
import re
import sys
import tomllib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

# The script filename would otherwise shadow the installed lightgbm package.
if not __package__:
    script_directory = Path(__file__).resolve().parent
    sys.path = [path for path in sys.path
                if Path(path or ".").resolve() != script_directory]
    sys.path.insert(0, str(script_directory.parents[1]))
    __package__ = "2026_10_09_factor_research.model"

import lightgbm as lgb

CONFIG_PATH = Path(__file__).with_name("config.toml")

# LightGBM training configuration; edit these globals to tune the model.
NUM_BOOST_ROUND = 100
LIGHTGBM_PARAMS = {
    "objective": "regression",
    "learning_rate": 0.05,
    "num_leaves": 15,
    "min_data_in_leaf": 1000,
    "num_threads": 4,
    "seed": 42,
    "deterministic": True,
    "force_col_wise": True,
    "verbosity": -1,
}


def horizon_ns(column: str) -> int:
    """Parse the fixed-duration labels emitted by factors/returns.py."""
    label = column.removeprefix("return_")
    if not re.fullmatch(r"(?:\d+(?:\.\d+)?(?:ns|us|ms|s|m|h|d|w))+", label):
        raise ValueError(f"Invalid return horizon: {column}")
    units = {"ns": 1, "us": 1_000, "ms": 1_000_000, "s": 10**9,
             "m": 60 * 10**9, "h": 3600 * 10**9,
             "d": 86400 * 10**9, "w": 604800 * 10**9}
    duration = sum(Fraction(value) * units[unit] for value, unit in
                   re.findall(r"(\d+(?:\.\d+)?)(ns|us|ms|s|m|h|d|w)", label))
    if duration <= 0 or duration.denominator != 1:
        raise ValueError(f"Invalid return horizon: {column}")
    return int(duration)


def average_ranks(values: np.ndarray) -> np.ndarray:
    """Rank each column, assigning average ranks to tied observations."""
    return pl.DataFrame(values).select(pl.all().rank(method="average")).to_numpy()


def rank_ic(feature_ranks: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Correlate feature ranks with target ranks on the same finite rows.

    Callers filter missing targets before ranking features. Constant columns
    and samples shorter than two observations produce NaN.
    """
    if len(target) < 2:
        return np.full(feature_ranks.shape[1], np.nan)
    x = feature_ranks
    y = pl.Series(target).rank(method="average").to_numpy()
    x = x - x.mean(axis=0)
    y = y - y.mean()
    denominator = np.sqrt(np.sum(x * x, axis=0) * np.sum(y * y))
    return np.divide(x.T @ y, denominator,
                     out=np.full(feature_ranks.shape[1], np.nan), where=denominator > 0)


def load_data(config_path: Path = CONFIG_PATH) -> tuple:
    """Read the configured symbol and split with a forward-return embargo."""
    config_path = Path(config_path).resolve()
    with config_path.open("rb") as source:
        config = tomllib.load(source)
    symbol = config["data"]["symbol"]
    fraction = config["model"].get("train_fraction", 0.7)
    if not 0 < fraction < 1:
        raise ValueError("model.train_fraction must be between 0 and 1")
    frame = pl.read_parquet(
        config_path.parent / config["data"]["input"] / f"{symbol}_factors.parquet"
    )
    if "timestamp" not in frame.columns or frame["timestamp"].null_count():
        raise ValueError("Input requires non-null timestamp values")
    frame = frame.sort("timestamp")
    # return_volume_ratio_* are input factors, not future-return labels.
    targets = [name for name in frame.columns if re.fullmatch(
        r"return_(?:\d+(?:\.\d+)?(?:ns|us|ms|s|m|h|d|w))+", name
    )]
    factors = [name for name in frame.columns
               if name != "timestamp" and name not in targets]
    if not factors or not targets:
        raise ValueError("Input requires factor columns and return_* targets")
    if any(not frame.schema[name].is_numeric() for name in factors + targets):
        raise ValueError("All factor and return columns must be numeric")
    split = int(frame.height * fraction)
    if split < 2 or frame.height - split < 2:
        raise ValueError("At least two training and two test rows are required")
    test_start = frame["timestamp"][split]
    # A shared embargo lets all horizons use the same training observations.
    embargo = max(horizon_ns(name) for name in targets)
    train = frame.filter(pl.col("timestamp") < test_start - embargo)
    test = frame.filter(pl.col("timestamp") >= test_start)
    finite_factors = pl.all_horizontal([pl.col(name).is_finite() for name in factors])
    train = train.filter(finite_factors & pl.all_horizontal(
        [pl.col(name).is_finite() for name in targets]
    ))
    test = test.filter(finite_factors)
    if train.height < 2 or test.height < 2:
        raise ValueError("Too few finite observations after filtering and embargo")
    output = config_path.parent / config["output"]["directory"]
    return train, test, factors, targets, symbol, fraction, embargo, output


def plot_rank_ic(test, predictions, factors, targets, symbol, fraction, embargo,
                 output) -> Path:
    """Write only a PNG of single-factor and combined test Rank IC."""
    test_x = test.select(factors).to_numpy()
    ic = np.empty((len(factors) + 1, len(targets)))
    previous_valid = None
    feature_ranks = None
    for index, target in enumerate(targets):
        actual = test[target].to_numpy()
        valid = np.isfinite(actual)
        # Reuse ranks for horizons with identical valid rows. If target rows
        # differ, re-rank the factors within that horizon's valid sample.
        if previous_valid is None or not np.array_equal(valid, previous_valid):
            feature_ranks = average_ranks(test_x[valid])
            previous_valid = valid
        ic[:-1, index] = rank_ic(feature_ranks, actual[valid])
        prediction_ranks = average_ranks(predictions[valid, index:index + 1])
        ic[-1, index] = rank_ic(prediction_ranks, actual[valid])[0]

    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{symbol}_lightgbm_ic.png"
    labels = factors + ["LIGHTGBM_COMBINED"]
    fig, ax = plt.subplots(figsize=(max(10, len(targets) * 0.8), len(labels) * 0.3 + 2))
    heatmap = ax.imshow(np.ma.masked_invalid(ic), cmap="RdBu_r", vmin=-1, vmax=1,
                        aspect="auto")
    ax.set_xticks(range(len(targets)), [name.removeprefix("return_") for name in targets],
                  rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.axhline(len(factors) - 0.5, color="black", linewidth=1)
    for row in range(len(labels)):
        for col in range(len(targets)):
            value = ic[row, col]
            ax.text(col, row, f"{value:.3f}" if np.isfinite(value) else "N/A",
                    ha="center", va="center", fontsize=7,
                    color="white" if abs(value) > 0.55 else "black")
    ax.set_title(f"{symbol} | LightGBM combination | test Rank IC (Spearman)\n"
                 f"Chronological train/test: {fraction:.0%}/{1-fraction:.0%}; "
                 f"embargo: {embargo / 10**9:g}s")
    ax.set_xlabel("Forward return horizon")
    fig.colorbar(heatmap, ax=ax, label="Rank IC (Spearman)", fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"Heatmap saved to {path.resolve()}")
    return path.resolve()


def run_lightgbm(config_path: Path = CONFIG_PATH) -> Path:
    """Train on the chronological training split; test data is evaluation only."""
    rounds = NUM_BOOST_ROUND
    if not isinstance(rounds, int) or isinstance(rounds, bool) or rounds <= 0:
        raise ValueError("NUM_BOOST_ROUND must be a positive integer")
    params = LIGHTGBM_PARAMS.copy()
    train, test, factors, targets, symbol, fraction, embargo, output = load_data(config_path)
    x = train.select(factors).to_numpy().astype(np.float32)
    y = train.select(targets).to_numpy()
    test_x = test.select(factors).to_numpy().astype(np.float32)
    # Share the binned feature dataset across horizons; only labels change.
    dataset = lgb.Dataset(x, label=y[:, 0], feature_name=factors, params=params)
    dataset.construct()
    del x, train
    predictions = np.empty((test.height, len(targets)))
    for index, target in enumerate(targets):
        dataset.set_label(y[:, index])
        booster = lgb.train(params, dataset, num_boost_round=rounds)
        predictions[:, index] = booster.predict(test_x, num_threads=params["num_threads"])
        print(f"[{index + 1}/{len(targets)}] {target}: trained {booster.current_iteration()} rounds",
              flush=True)
        del booster
    del dataset, y, test_x
    return plot_rank_ic(test, predictions, factors, targets, symbol, fraction,
                        embargo, output)


if __name__ == "__main__":
    run_lightgbm()
