"""Volume-weighted trade price deviation from the contemporaneous BBO midpoint."""

from datetime import timedelta
import re

import polars as pl


def generate_price_versus_mid_factor(
    factors: pl.DataFrame,
    bbo: pl.LazyFrame,
    trade: pl.LazyFrame,
    depth: pl.LazyFrame,
    period: str | timedelta,
) -> pl.DataFrame:
    """Append ``price_versus_mid_{period}`` at each requested timestamp.

    Match each trade to the last BBO at or before its recv_ns, then compute
    log(px / mid), where mid = (bid_px + ask_px) / 2. Equal BBO timestamps
    use the last input row. Sum qty * log(px / mid) in (t-period, t] and
    divide by total qty in that window. All trades, including trades at
    equal timestamps, contribute. No future BBO is used.

    History starts at the later of the first BBO and first trade timestamps.
    Incomplete windows, empty streams and null targets yield null; complete
    windows with zero volume yield 0.0. Inputs must be for one symbol with
    Int64 ns recv_ns/timestamp, positive non-null prices in consistent units,
    and nonnegative non-null qty. Fixed price/quantity scaling cancels.
    Period accepts positive fixed durations (ns/us/ms/s/m/h/d/w) or timedelta.
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

    factor_column = f"price_versus_mid_{label}"
    targets = (
        factors.select(pl.col("timestamp").cast(pl.Datetime("ns")))
        .lazy().with_row_index("_row_index")
    )
    quotes = (
        bbo.select(
            pl.col("recv_ns").cast(pl.Datetime("ns")).alias("timestamp"),
            ((pl.col("bid_px").cast(pl.Float64) + pl.col("ask_px").cast(pl.Float64)) / 2)
            .alias("_mid"),
        )
        .drop_nulls("timestamp")
        .sort("timestamp", maintain_order=True)
    )
    trades = (
        trade.select(
            pl.col("recv_ns").cast(pl.Datetime("ns")).alias("timestamp"),
            pl.col("px").cast(pl.Float64).alias("_price"),
            pl.col("qty").cast(pl.Float64).alias("_volume"),
        )
        .drop_nulls("timestamp")
        .sort("timestamp")
    )
    priced_trades = trades.join_asof(quotes, on="timestamp", strategy="backward").select(
        "timestamp", "_volume",
        (pl.col("_volume") * (pl.col("_price") / pl.col("_mid")).log())
        .alias("_weighted_deviation"),
    )
    target_events = (
        targets.select("timestamp").drop_nulls("timestamp").unique()
        .with_columns(
            pl.lit(0.0).alias("_volume"),
            pl.lit(0.0).alias("_weighted_deviation"),
        )
    )
    windows = (
        pl.concat([priced_trades, target_events])
        .group_by("timestamp")
        .agg(pl.col("_volume").sum(), pl.col("_weighted_deviation").sum())
        .sort("timestamp")
        .rolling("timestamp", period=f"{period_ns}ns", closed="right")
        .agg(pl.col("_volume").sum(), pl.col("_weighted_deviation").sum())
    )
    values = (
        targets.join(windows, on="timestamp", how="left")
        .join(quotes.select(pl.col("timestamp").min().alias("_bbo_start")), how="cross")
        .join(trades.select(pl.col("timestamp").min().alias("_trade_start")), how="cross")
        .sort("_row_index")
        .select(
            pl.when(
                pl.col("timestamp").is_null()
                | pl.col("_bbo_start").is_null()
                | pl.col("_trade_start").is_null()
                | ((pl.col("timestamp") - pl.col("_bbo_start"))
                   .dt.total_nanoseconds() < period_ns)
                | ((pl.col("timestamp") - pl.col("_trade_start"))
                   .dt.total_nanoseconds() < period_ns)
            )
            .then(None)
            .when(pl.col("_volume") == 0)
            .then(0.0)
            .otherwise(pl.col("_weighted_deviation") / pl.col("_volume"))
            .alias(factor_column)
        )
        .collect()
    )
    return factors.with_columns(values.get_columns())
