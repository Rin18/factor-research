"""Trailing BBO midpoint return divided by trade volume."""

from datetime import timedelta
import re

import polars as pl


def generate_return_volume_ratio_factor(
    factors: pl.DataFrame,
    bbo: pl.LazyFrame,
    trade: pl.LazyFrame,
    depth: pl.LazyFrame,
    period: str | timedelta,
) -> pl.DataFrame:
    """Append ``return_volume_ratio_{period}`` at the requested timestamps.

    Compute (mid(t) / mid(t - period) - 1) / sum(qty), with trade volume
    over (t - period, t]. Midpoints use the last BBO received at or before
    each endpoint; equal BBO timestamps use the last input row. All trades
    sharing a timestamp contribute to volume. No complete history in either
    stream yields null; a complete window with zero volume yields 0.0.
    History starts at the later of the first BBO and first trade timestamps.
    Empty streams and null target timestamps also lack sufficient history.

    Inputs must be for one symbol with Int64 nanosecond recv_ns/timestamp,
    positive, non-null BBO prices and nonnegative, non-null trade qty.
    Volume uses the supplied qty units (including any fixed-point scaling).
    Period accepts a positive fixed duration (ns/us/ms/s/m/h/d/w) or timedelta.
    Preserve input rows, order and other columns; replace an existing output
    column. Depth is accepted for the common interface and is not evaluated.
    """
    duration_units = {
        "ns": 1, "us": 1_000, "ms": 1_000_000, "s": 1_000_000_000,
        "m": 60_000_000_000, "h": 3_600_000_000_000,
        "d": 86_400_000_000_000, "w": 604_800_000_000_000,
    }
    if isinstance(period, timedelta):
        microseconds = period // timedelta(microseconds=1)
        period_ns = microseconds * 1_000
        label = f"{microseconds}us"
    elif isinstance(period, str) and re.fullmatch(
        r"(?:\d+(?:ns|us|ms|s|m|h|d|w))+", period
    ):
        period_ns = sum(
            int(value) * duration_units[unit]
            for value, unit in re.findall(r"(\d+)(ns|us|ms|s|m|h|d|w)", period)
        )
        label = period
    else:
        raise ValueError("period must be a positive fixed duration string or timedelta")
    if not 0 < period_ns <= 2**63 - 1:
        raise ValueError("period must be positive and fit in Int64 nanoseconds")

    factor_column = f"return_volume_ratio_{label}"
    targets = factors.select("timestamp").lazy().with_row_index("_row_index")
    quotes = (
        bbo.select(
            pl.col("recv_ns").alias("timestamp"),
            ((pl.col("bid_px").cast(pl.Float64) + pl.col("ask_px").cast(pl.Float64)) / 2)
            .alias("_mid"),
        )
        .drop_nulls("timestamp")
        .sort("timestamp", maintain_order=True)
    )
    trades = trade.select(
        pl.col("recv_ns").alias("timestamp"),
        pl.col("qty").cast(pl.Float64).alias("_volume"),
    ).drop_nulls("timestamp")
    target_events = (
        targets.select("timestamp").drop_nulls("timestamp").unique()
        .with_columns(pl.lit(0.0).alias("_volume"))
    )
    volumes = (
        pl.concat([trades, target_events])
        .group_by("timestamp").agg(pl.col("_volume").sum())
        .sort("timestamp")
        .with_columns(pl.col("timestamp").cast(pl.Datetime("ns")))
        .rolling("timestamp", period=f"{period_ns}ns", closed="right")
        .agg(pl.col("_volume").sum())
        .with_columns(pl.col("timestamp").cast(pl.Int64))
    )
    values = (
        targets.join(volumes, on="timestamp", how="left")
        .join(quotes.select(pl.col("timestamp").min().alias("_bbo_start")), how="cross")
        .join(trades.select(pl.col("timestamp").min().alias("_trade_start")), how="cross")
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
                pl.col("_past_ns").is_null()
                | pl.col("_bbo_start").is_null()
                | pl.col("_trade_start").is_null()
                | (pl.col("_past_ns") < pl.col("_bbo_start"))
                | (pl.col("_past_ns") < pl.col("_trade_start"))
            )
            .then(None)
            .when(pl.col("_volume") == 0)
            .then(0.0)
            .otherwise((pl.col("_mid") / pl.col("_past_mid") - 1) / pl.col("_volume"))
            .alias(factor_column)
        )
        .collect()
    )
    return factors.with_columns(values.get_columns())
