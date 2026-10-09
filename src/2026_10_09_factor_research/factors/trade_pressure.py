"""Trade notional and count pressure over short and reference windows."""

from datetime import timedelta
import re

import polars as pl


def generate_trade_pressure_factor(
    factors: pl.DataFrame,
    bbo: pl.LazyFrame,
    trade: pl.LazyFrame,
    depth: pl.LazyFrame,
    pressure_period: str | timedelta,
    reference_period: str | timedelta,
) -> pl.DataFrame:
    """Append ``trade_pressure_{pressure_period}_{reference_period}``.

    Numerator: aggressive sell notional minus aggressive buy notional in
    (timestamp - pressure_period, timestamp]. Denominator: total notional
    in (timestamp - reference_period, timestamp] multiplied by
    pressure_period / reference_period, i.e. the average notional per
    pressure period. Positive values indicate sell pressure.

    Notional is px * qty; is_buyer_maker=True indicates an aggressive sell.
    Inputs must be for one symbol, with non-null px, qty, is_buyer_maker,
    and Int64 nanosecond timestamps (factors.timestamp and trade.recv_ns).
    Price and quantity scales must be consistent across all trades; their
    fixed-point scaling cancels in the ratio.

    Periods must be positive fixed durations: strings such as "5s", "1m",
    "1h30m" (ns/us/ms/s/m/h/d/w), or timedelta. A day means 24 hours.
    No reference notional yields null. No trades in the pressure period
    yields zero when the denominator is positive. Preserve input rows, order, and other
    columns without mutation; replace the output column if it exists.
    BBO and depth are accepted for the common interface and not evaluated.
    """
    duration_units = {
        "ns": 1,
        "us": 1_000,
        "ms": 1_000_000,
        "s": 1_000_000_000,
        "m": 60_000_000_000,
        "h": 3_600_000_000_000,
        "d": 86_400_000_000_000,
        "w": 604_800_000_000_000,
    }
    period_ns = []
    period_labels = []
    for name, period in (
        ("pressure_period", pressure_period),
        ("reference_period", reference_period),
    ):
        if isinstance(period, timedelta):
            microseconds = period // timedelta(microseconds=1)
            nanoseconds = microseconds * 1_000
            label = f"{microseconds}us"
        elif isinstance(period, str) and re.fullmatch(
            r"(?:\d+(?:ns|us|ms|s|m|h|d|w))+", period
        ):
            nanoseconds = sum(
                int(value) * duration_units[unit]
                for value, unit in re.findall(r"(\d+)(ns|us|ms|s|m|h|d|w)", period)
            )
            label = period
        else:
            raise ValueError(f"{name} must be a positive fixed duration string or timedelta")
        if not 0 < nanoseconds <= 2**63 - 1:
            raise ValueError(f"{name} must be positive and fit in Int64 nanoseconds")
        period_ns.append(nanoseconds)
        period_labels.append(label)

    pressure_period_ns, reference_period_ns = period_ns
    pressure_label, reference_label = period_labels
    trade_timestamp_column = "recv_ns"
    factor_column = f"trade_pressure_{pressure_label}_{reference_label}"
    targets = (
        factors.select(pl.col("timestamp").cast(pl.Datetime("ns")))
        .lazy()
        .with_row_index("_row_index")
    )
    notional = pl.col("px").cast(pl.Float64) * pl.col("qty").cast(pl.Float64)
    trades = trade.select(
        pl.col(trade_timestamp_column).cast(pl.Datetime("ns")).alias("timestamp"),
        pl.when(pl.col("is_buyer_maker"))
        .then(notional)
        .otherwise(-notional)
        .alias("_signed_notional"),
        notional.alias("_total_notional"),
    ).drop_nulls("timestamp")
    target_events = (
        targets.select("timestamp")
        .drop_nulls("timestamp")
        .unique()
        .with_columns(
            pl.lit(0.0).alias("_signed_notional"),
            pl.lit(0.0).alias("_total_notional"),
        )
    )
    events = (
        pl.concat([trades, target_events])
        .group_by("timestamp")
        .agg(pl.col("_signed_notional").sum(), pl.col("_total_notional").sum())
        .sort("timestamp")
    )
    numerator = events.rolling(
        "timestamp", period=f"{pressure_period_ns}ns", closed="right"
    ).agg(
        pl.col("_signed_notional").sum()
    )
    denominator = events.rolling(
        "timestamp", period=f"{reference_period_ns}ns", closed="right"
    ).agg(
        (pl.col("_total_notional").sum() * (pressure_period_ns / reference_period_ns))
        .alias("_mean_notional")
    )
    pressure = (
        targets.join(numerator, on="timestamp", how="left")
        .join(denominator, on="timestamp", how="left")
        .sort("_row_index")
        .select(
            pl.when(pl.col("_mean_notional") != 0)
            .then(pl.col("_signed_notional") / pl.col("_mean_notional"))
            .otherwise(None)
            .alias(factor_column)
        )
        .collect()
        .get_column(factor_column)
    )
    return factors.with_columns(pressure)


def generate_trade_count_pressure_factor(
    factors: pl.DataFrame,
    bbo: pl.LazyFrame,
    trade: pl.LazyFrame,
    depth: pl.LazyFrame,
    pressure_period: str | timedelta,
    reference_period: str | timedelta,
) -> pl.DataFrame:
    """Append ``trade_count_pressure_{pressure_period}_{reference_period}``.

    Numerator: aggressive sell count minus aggressive buy count in
    (timestamp - pressure_period, timestamp]. Denominator: total trade count
    in (timestamp - reference_period, timestamp] multiplied by
    pressure_period / reference_period. Positive values indicate sell pressure.

    Each trade row counts once, including trades sharing a timestamp.
    is_buyer_maker=True indicates an aggressive sell; False indicates a buy.
    Quantity and price are not used. Inputs must be for one symbol, with
    non-null is_buyer_maker and Int64 nanosecond timestamps
    (factors.timestamp and trade.recv_ns).

    Periods must be positive fixed durations: strings such as "5s", "1m",
    "1h30m" (ns/us/ms/s/m/h/d/w), or timedelta. A day means 24 hours.
    No reference trades yields null. No trades in the pressure period yields
    zero when the reference count is positive. Preserve input rows, order,
    and other columns without mutation; replace the output column if it
    exists. BBO and depth are accepted for the common interface and not evaluated.
    """
    duration_units = {
        "ns": 1,
        "us": 1_000,
        "ms": 1_000_000,
        "s": 1_000_000_000,
        "m": 60_000_000_000,
        "h": 3_600_000_000_000,
        "d": 86_400_000_000_000,
        "w": 604_800_000_000_000,
    }
    period_ns = []
    period_labels = []
    for name, period in (
        ("pressure_period", pressure_period),
        ("reference_period", reference_period),
    ):
        if isinstance(period, timedelta):
            microseconds = period // timedelta(microseconds=1)
            nanoseconds = microseconds * 1_000
            label = f"{microseconds}us"
        elif isinstance(period, str) and re.fullmatch(
            r"(?:\d+(?:ns|us|ms|s|m|h|d|w))+", period
        ):
            nanoseconds = sum(
                int(value) * duration_units[unit]
                for value, unit in re.findall(r"(\d+)(ns|us|ms|s|m|h|d|w)", period)
            )
            label = period
        else:
            raise ValueError(f"{name} must be a positive fixed duration string or timedelta")
        if not 0 < nanoseconds <= 2**63 - 1:
            raise ValueError(f"{name} must be positive and fit in Int64 nanoseconds")
        period_ns.append(nanoseconds)
        period_labels.append(label)

    pressure_period_ns, reference_period_ns = period_ns
    pressure_label, reference_label = period_labels
    trade_timestamp_column = "recv_ns"
    factor_column = f"trade_count_pressure_{pressure_label}_{reference_label}"
    targets = (
        factors.select(pl.col("timestamp").cast(pl.Datetime("ns")))
        .lazy()
        .with_row_index("_row_index")
    )
    trades = trade.select(
        pl.col(trade_timestamp_column).cast(pl.Datetime("ns")).alias("timestamp"),
        pl.when(pl.col("is_buyer_maker"))
        .then(pl.lit(1.0))
        .otherwise(pl.lit(-1.0))
        .alias("_signed_count"),
        pl.lit(1.0).alias("_total_count"),
    ).drop_nulls("timestamp")
    target_events = (
        targets.select("timestamp")
        .drop_nulls("timestamp")
        .unique()
        .with_columns(
            pl.lit(0.0).alias("_signed_count"),
            pl.lit(0.0).alias("_total_count"),
        )
    )
    events = (
        pl.concat([trades, target_events])
        .group_by("timestamp")
        .agg(pl.col("_signed_count").sum(), pl.col("_total_count").sum())
        .sort("timestamp")
    )
    numerator = events.rolling(
        "timestamp", period=f"{pressure_period_ns}ns", closed="right"
    ).agg(
        pl.col("_signed_count").sum()
    )
    denominator = events.rolling(
        "timestamp", period=f"{reference_period_ns}ns", closed="right"
    ).agg(
        (pl.col("_total_count").sum() * (pressure_period_ns / reference_period_ns))
        .alias("_mean_count")
    )
    pressure = (
        targets.join(numerator, on="timestamp", how="left")
        .join(denominator, on="timestamp", how="left")
        .sort("_row_index")
        .select(
            pl.when(pl.col("_mean_count") != 0)
            .then(pl.col("_signed_count") / pl.col("_mean_count"))
            .otherwise(None)
            .alias(factor_column)
        )
        .collect()
        .get_column(factor_column)
    )
    return factors.with_columns(pressure)
