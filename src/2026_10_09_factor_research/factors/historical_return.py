"""Trailing returns from BBO midpoint prices."""

from datetime import timedelta
from fractions import Fraction
import re

import polars as pl


def generate_historical_return_factor(
    factors: pl.DataFrame,
    bbo: pl.LazyFrame,
    trade: pl.LazyFrame,
    depth: pl.LazyFrame,
    period: str | timedelta,
) -> pl.DataFrame:
    """Append ``historical_return_{period}`` = mid(t) / mid(t-period) - 1.

    Use the last BBO received at or before each endpoint, never a future
    quote. Equal BBO timestamps use the last input row. Midpoint is
    (bid_px + ask_px) / 2; fixed price scaling cancels in the ratio.
    Missing history, null timestamps, invalid/nonpositive midpoint prices,
    and Int64 underflow yield null. Preserve rows, order and other columns;
    replace an existing output column of the same name.

    Period accepts a positive fixed duration (ns/us/ms/s/m/h/d/w), including
    decimals or compound durations, or timedelta. Timestamps must be Int64
    nanoseconds and data must be for one symbol. Trade/depth are accepted for
    the standard factor interface and are not evaluated.
    """
    units = {
        "ns": 1, "us": 1_000, "ms": 1_000_000, "s": 1_000_000_000,
        "m": 60_000_000_000, "h": 3_600_000_000_000,
        "d": 86_400_000_000_000, "w": 604_800_000_000_000,
    }
    if isinstance(period, timedelta):
        microseconds = period // timedelta(microseconds=1)
        period_ns = microseconds * 1_000
        label = f"{microseconds}us"
    elif isinstance(period, str) and re.fullmatch(
        r"(?:\d+(?:\.\d+)?(?:ns|us|ms|s|m|h|d|w))+", period
    ):
        duration = sum(
            Fraction(value) * units[unit]
            for value, unit in re.findall(
                r"(\d+(?:\.\d+)?)(ns|us|ms|s|m|h|d|w)", period
            )
        )
        if duration.denominator != 1:
            raise ValueError("period must resolve to whole nanoseconds")
        period_ns = int(duration)
        label = period
    else:
        raise ValueError("period must be a positive fixed duration string or timedelta")
    if not 0 < period_ns <= 2**63 - 1:
        raise ValueError("period must be positive and fit in Int64 nanoseconds")

    quotes = (
        bbo.select(
            pl.col("recv_ns").alias("timestamp"),
            ((pl.col("bid_px").cast(pl.Float64) + pl.col("ask_px").cast(pl.Float64)) / 2)
            .alias("_mid"),
        )
        .drop_nulls("timestamp")
        .sort("timestamp", maintain_order=True)
    )
    values = (
        factors.select("timestamp").lazy().with_row_index("_row_index")
        .with_columns(
            pl.when(pl.col("timestamp") >= -(2**63) + period_ns)
            .then(pl.col("timestamp") - period_ns)
            .otherwise(None)
            .alias("_past_ns")
        )
        .sort("timestamp")
        .join_asof(quotes, on="timestamp", strategy="backward")
        .sort("_past_ns")
        .join_asof(
            quotes.select(
                pl.col("timestamp").alias("_past_ns"),
                pl.col("_mid").alias("_past_mid"),
            ),
            on="_past_ns", strategy="backward",
        )
        .sort("_row_index")
        .select(
            pl.when(
                (pl.col("_mid") > 0) & pl.col("_mid").is_finite()
                & (pl.col("_past_mid") > 0) & pl.col("_past_mid").is_finite()
            )
            .then(pl.col("_mid") / pl.col("_past_mid") - 1)
            .otherwise(None)
            .alias(f"historical_return_{label}")
        )
        .collect()
    )
    return factors.with_columns(values.get_columns())
