"""Read recorder Parquet files and generate a factor DataFrame.

Run from the project root:
    uv run python src/2026_10_09_factor_research/factors/build_factors.py
or:
    uv run python -m 2026_10_09_factor_research.factors.build_factors

Read the adjacent config.toml. Paths inside the configuration are relative
to that file. Timestamps remain Int64 nanoseconds.
The CLI prints only the generated DataFrame; build_factors returns it.
"""

import tomllib
from pathlib import Path

import polars as pl

# Allow relative imports when this file is run directly in the project environment.
__package__ = "2026_10_09_factor_research.factors"

from .obi import generate_multi_obi_factor, generate_obi_factor
from .trade_imbalance import (
    generate_trade_count_imbalance_factor,
    generate_trade_imbalance_factor,
)
from .trade_pressure import (
    generate_trade_count_pressure_factor,
    generate_trade_pressure_factor,
)

CONFIG_PATH = Path(__file__).with_name("config.toml")


def build_factors() -> pl.DataFrame:
    """Load one symbol and compute factors on a configured regular time grid.

    Start/end are inclusive and limited to the three streams' recv_ns overlap.
    Grid points are multiples of interval_ns since the Unix epoch. Keep all
    source history so quotes and trade windows preceding start remain usable.
    """
    with CONFIG_PATH.open("rb") as source:
        config = tomllib.load(source)

    data_config = config["data"]
    input_path = CONFIG_PATH.parent / data_config["input"]
    symbol = data_config["symbol"]
    bbo = pl.scan_parquet(input_path / "bbo" / f"{symbol}.parquet")
    trade = pl.scan_parquet(input_path / "trade" / f"{symbol}.parquet")
    depth = pl.scan_parquet(input_path / "depth" / f"{symbol}.parquet")

    time_config = config["timestamps"]
    interval_ns = time_config["interval_ns"]
    if interval_ns <= 0:
        raise ValueError("timestamps.interval_ns must be positive")
    bounds = pl.collect_all(
        [
            frame.select(
                pl.col("recv_ns").min().alias("start"),
                pl.col("recv_ns").max().alias("end"),
            )
            for frame in (bbo, trade, depth)
        ]
    )
    if any(bound["start"][0] is None for bound in bounds):
        raise ValueError("Cannot determine time overlap: a data stream is empty")
    overlap_start_ns = max(bound["start"][0] for bound in bounds)
    overlap_end_ns = min(bound["end"][0] for bound in bounds)
    start_ns = max(time_config.get("start_ns", overlap_start_ns), overlap_start_ns)
    end_ns = min(time_config.get("end_ns", overlap_end_ns), overlap_end_ns)
    if start_ns > end_ns:
        raise ValueError("No common time range within the configured timestamps")
    first_ns = -(-start_ns // interval_ns) * interval_ns
    factors = pl.DataFrame(
        {
            "timestamp": pl.Series(
                range(first_ns, end_ns + 1, interval_ns), dtype=pl.Int64
            ),
        }
    )
    return generate_factors(factors, bbo, trade, depth)


def generate_factors(
    factors: pl.DataFrame,
    bbo: pl.LazyFrame,
    trade: pl.LazyFrame,
    depth: pl.LazyFrame,
) -> pl.DataFrame:
    """Append factor columns at the input timestamps and return the DataFrame."""
    factors = generate_obi_factor(factors, bbo, trade, depth)
    for levels in (5, 10, 20):
        factors = generate_multi_obi_factor(factors, bbo, trade, depth, levels=levels)
    for period in ("5s", "1m"):
        factors = generate_trade_imbalance_factor(
            factors,
            bbo,
            trade,
            depth,
            history_period=period,
        )
    for period in ("5s", "1m"):
        factors = generate_trade_count_imbalance_factor(
            factors,
            bbo,
            trade,
            depth,
            history_period=period,
        )
    factors = generate_trade_pressure_factor(
        factors,
        bbo,
        trade,
        depth,
        pressure_period="5s",
        reference_period="1m",
    )
    factors = generate_trade_count_pressure_factor(
        factors,
        bbo,
        trade,
        depth,
        pressure_period="5s",
        reference_period="1m",
    )
    return factors


def main() -> None:
    print(build_factors())


if __name__ == "__main__":
    main()
