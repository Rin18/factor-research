"""Forward returns from BBO midpoint prices."""

from datetime import timedelta
from fractions import Fraction
import re

import polars as pl


def generate_return(
    factors: pl.DataFrame,
    bbo: pl.LazyFrame,
    future_periods: list[str | timedelta],
) -> pl.DataFrame:
    """Append ``return_{period}`` columns without changing input row order.

    Return is mid(t + period) / mid(t) - 1, where mid = (bid_px + ask_px) / 2.
    Use the last BBO received at or before each price observation time;
    equal timestamps use the last input row. Timestamps must be Int64 ns
    and data must be for one symbol. Periods are positive fixed durations
    (ns/us/ms/s/m/h/d/w), including decimals such as "2.5s", or timedelta;
    a day means 24 hours. Durations must resolve to whole nanoseconds.

    Missing prices, non-positive midpoint prices, null target timestamps,
    and observation times beyond the last BBO timestamp yield null.
    Preserve other columns and replace existing return columns of the same
    name. Price scaling cancels in the ratio. Factors is not mutated.
    """
    if not future_periods:
        return factors

    duration_units = {
        "ns": 1, "us": 1_000, "ms": 1_000_000, "s": 1_000_000_000,
        "m": 60_000_000_000, "h": 3_600_000_000_000,
        "d": 86_400_000_000_000, "w": 604_800_000_000_000,
    }
    periods = {}
    for period in future_periods:
        if isinstance(period, timedelta):
            microseconds = period // timedelta(microseconds=1)
            nanoseconds = microseconds * 1_000
            label = f"{microseconds}us"
        elif isinstance(period, str) and re.fullmatch(
            r"(?:\d+(?:\.\d+)?(?:ns|us|ms|s|m|h|d|w))+", period
        ):
            nanoseconds = sum(
                Fraction(value) * duration_units[unit]
                for value, unit in re.findall(
                    r"(\d+(?:\.\d+)?)(ns|us|ms|s|m|h|d|w)", period
                )
            )
            if nanoseconds.denominator != 1:
                raise ValueError(f"future period {period!r} must resolve to whole nanoseconds")
            nanoseconds = int(nanoseconds)
            label = period
        else:
            raise ValueError(f"future period {period!r} must be a positive fixed duration")
        if not 0 < nanoseconds <= 2**63 - 1:
            raise ValueError(f"future period {period!r} must be positive and fit in Int64 ns")
        periods[f"return_{label}"] = nanoseconds

    bbo_timestamp_column = "recv_ns"
    quotes = (
        bbo.select(
            pl.col(bbo_timestamp_column).alias("timestamp"),
            ((pl.col("bid_px").cast(pl.Float64) + pl.col("ask_px").cast(pl.Float64)) / 2)
            .alias("_mid"),
        )
        .drop_nulls("timestamp")
        .sort("timestamp", maintain_order=True)
        .with_columns(pl.col("timestamp").max().alias("_bbo_end_ns"))
    )
    values = (
        factors.select("timestamp")
        .lazy()
        .with_row_index("_row_index")
        .sort("timestamp")
        .join_asof(quotes, on="timestamp", strategy="backward")
    )
    for column, nanoseconds in periods.items():
        values = (
            values.with_columns(
                pl.when(pl.col("timestamp") <= 2**63 - 1 - nanoseconds)
                .then(pl.col("timestamp") + nanoseconds)
                .otherwise(None)
                .alias("_future_ns")
            )
            .sort("_future_ns")
            .join_asof(
                quotes.select(
                    pl.col("timestamp").alias("_future_ns"),
                    pl.col("_mid").alias("_future_mid"),
                ),
                on="_future_ns",
                strategy="backward",
            )
            .with_columns(
                pl.when(
                    (pl.col("_future_ns") <= pl.col("_bbo_end_ns"))
                    & (pl.col("_mid") > 0)
                    & (pl.col("_future_mid") > 0)
                )
                .then(pl.col("_future_mid") / pl.col("_mid") - 1)
                .otherwise(None)
                .alias(column)
            )
            .drop("_future_ns", "_future_mid")
        )
    returns = values.sort("_row_index").select(list(periods)).collect()
    return factors.with_columns(returns.get_columns())
