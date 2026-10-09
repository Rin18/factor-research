"""Trade volume and count imbalance over a trailing time window."""

from datetime import timedelta

import polars as pl


def generate_trade_imbalance_factor(
    factors: pl.DataFrame,
    bbo: pl.LazyFrame,
    trade: pl.LazyFrame,
    depth: pl.LazyFrame,
    history_period: str | timedelta,
) -> pl.DataFrame:
    """Append ``trade_imbalance_{period}`` at each requested timestamp.

    ``history_period`` is a positive Polars duration such as "5s", "1m",
    or "1h", or a timedelta. Inputs must be for one symbol, with
    ``factors.timestamp`` and ``trade.recv_ns`` as Int64 nanoseconds.
    Trade also needs ``qty`` and ``is_buyer_maker`` (non-null).

    Sum quantities in (timestamp - history_period, timestamp]. A buyer
    maker implies an aggressive sell; otherwise the trade is an aggressive
    buy. Compute (buy_qty - sell_qty) / (buy_qty + sell_qty). Windows with
    no volume yield null. Trades sharing a timestamp are all included.

    Preserve original row order and other columns without mutating factors;
    replace the output column if it exists. BBO and depth are accepted for
    the common factor interface and are not evaluated.
    """
    if isinstance(history_period, timedelta):
        if history_period <= timedelta(0):
            raise ValueError("history_period must be positive")
        period_label = f"{history_period // timedelta(microseconds=1)}us"
    elif isinstance(history_period, str) and history_period:
        period_label = history_period
    else:
        raise ValueError("history_period must be a duration string or timedelta")

    trade_timestamp_column = "recv_ns"
    factor_column = f"trade_imbalance_{period_label}"
    targets = (
        factors.select(pl.col("timestamp").cast(pl.Datetime("ns")))
        .lazy()
        .with_row_index("_row_index")
    )
    qty = pl.col("qty").cast(pl.Float64)
    trades = trade.select(
        pl.col(trade_timestamp_column).cast(pl.Datetime("ns")).alias("timestamp"),
        pl.when(pl.col("is_buyer_maker"))
        .then(-qty)
        .otherwise(qty)
        .alias("_signed_qty"),
        qty.alias("_total_qty"),
    ).drop_nulls("timestamp")
    # Add zero-volume targets so windows are evaluated at the requested
    # times, including targets with no trade at that exact timestamp.
    target_events = targets.select("timestamp").drop_nulls("timestamp").unique().with_columns(
        pl.lit(0.0).alias("_signed_qty"),
        pl.lit(0.0).alias("_total_qty"),
    )
    windows = (
        pl.concat([trades, target_events])
        .group_by("timestamp")
        .agg(pl.col("_signed_qty").sum(), pl.col("_total_qty").sum())
        .sort("timestamp")
        .rolling("timestamp", period=history_period, closed="right")
        .agg(pl.col("_signed_qty").sum(), pl.col("_total_qty").sum())
        .select(
            "timestamp",
            pl.when(pl.col("_total_qty") != 0)
            .then(pl.col("_signed_qty") / pl.col("_total_qty"))
            .otherwise(None)
            .alias(factor_column),
        )
    )
    imbalance = (
        targets.join(windows, on="timestamp", how="left")
        .sort("_row_index")
        .select(factor_column)
        .collect()
        .get_column(factor_column)
    )
    return factors.with_columns(imbalance)


def generate_trade_count_imbalance_factor(
    factors: pl.DataFrame,
    bbo: pl.LazyFrame,
    trade: pl.LazyFrame,
    depth: pl.LazyFrame,
    history_period: str | timedelta,
) -> pl.DataFrame:
    """Append ``trade_count_imbalance_{period}`` using trade row counts.

    In (timestamp - history_period, timestamp], count aggressive buys
    (is_buyer_maker=False) and sells (True). Compute (buy_count - sell_count)
    / (buy_count + sell_count); no trades yields null. Each trade row counts
    once, including trades sharing a timestamp; quantity is not used.

    Inputs must be for one symbol. ``factors.timestamp`` and ``trade.recv_ns``
    use Int64 nanoseconds; ``is_buyer_maker`` must be non-null. History period
    accepts positive Polars duration strings (e.g. "5s") or timedelta.
    Preserve original rows, order, and other columns without mutation;
    replace the output column if it exists. BBO and depth are not evaluated.
    """
    if isinstance(history_period, timedelta):
        if history_period <= timedelta(0):
            raise ValueError("history_period must be positive")
        period_label = f"{history_period // timedelta(microseconds=1)}us"
    elif isinstance(history_period, str) and history_period:
        period_label = history_period
    else:
        raise ValueError("history_period must be a duration string or timedelta")

    trade_timestamp_column = "recv_ns"
    factor_column = f"trade_count_imbalance_{period_label}"
    targets = (
        factors.select(pl.col("timestamp").cast(pl.Datetime("ns")))
        .lazy()
        .with_row_index("_row_index")
    )
    trades = trade.select(
        pl.col(trade_timestamp_column).cast(pl.Datetime("ns")).alias("timestamp"),
        pl.when(pl.col("is_buyer_maker"))
        .then(pl.lit(-1.0))
        .otherwise(pl.lit(1.0))
        .alias("_signed_count"),
        pl.lit(1.0).alias("_total_count"),
    ).drop_nulls("timestamp")
    # Add zero-count targets so windows are evaluated at the requested
    # times, including targets with no trade at that exact timestamp.
    target_events = targets.select("timestamp").drop_nulls("timestamp").unique().with_columns(
        pl.lit(0.0).alias("_signed_count"),
        pl.lit(0.0).alias("_total_count"),
    )
    windows = (
        pl.concat([trades, target_events])
        .group_by("timestamp")
        .agg(pl.col("_signed_count").sum(), pl.col("_total_count").sum())
        .sort("timestamp")
        .rolling("timestamp", period=history_period, closed="right")
        .agg(pl.col("_signed_count").sum(), pl.col("_total_count").sum())
        .select(
            "timestamp",
            pl.when(pl.col("_total_count") != 0)
            .then(pl.col("_signed_count") / pl.col("_total_count"))
            .otherwise(None)
            .alias(factor_column),
        )
    )
    imbalance = (
        targets.join(windows, on="timestamp", how="left")
        .sort("_row_index")
        .select(factor_column)
        .collect()
        .get_column(factor_column)
    )
    return factors.with_columns(imbalance)
