"""RPI/non-RPI volume imbalance and their unweighted divergence."""

from datetime import timedelta
import re

import polars as pl


def generate_rpi_factors(
    factors: pl.DataFrame,
    bbo: pl.LazyFrame,
    trade: pl.LazyFrame,
    depth: pl.LazyFrame,
    history_period: str | timedelta,
) -> pl.DataFrame:
    """Append ``rpi_imbalance``, ``non_rpi_imbalance``, ``divergence``.

    Each output name is suffixed with ``_{period}``. Sum signed quantity
    divided by quantity separately for RPI and non-RPI trades in
    (timestamp - history_period, timestamp]. ``is_buyer_maker=False``
    means an aggressive buy (+1); True means an aggressive sell (-1).
    Divergence is non-RPI imbalance minus RPI imbalance, without weighting.
    A group with zero volume yields 0.0; divergence uses those values.

    Inputs must be for one symbol, with Int64 ns ``factors.timestamp`` and
    ``trade.recv_ns``, nonnegative non-null ``qty``, and non-null boolean
    ``is_rpi``/``is_buyer_maker``. Period accepts positive fixed durations
    (ns/us/ms/s/m/h/d/w) or timedelta. No future trades are used; available
    history is used even when the window precedes the start of the stream.
    All trades sharing a timestamp contribute. Null targets yield null.
    Preserve input rows, order and other columns without mutation; replace
    existing output columns. BBO and depth are accepted but not evaluated.
    """
    duration_units = {
        "ns": 1, "us": 1_000, "ms": 1_000_000, "s": 1_000_000_000,
        "m": 60_000_000_000, "h": 3_600_000_000_000,
        "d": 86_400_000_000_000, "w": 604_800_000_000_000,
    }
    if isinstance(history_period, timedelta):
        microseconds = history_period // timedelta(microseconds=1)
        period_ns = microseconds * 1_000
        label = f"{microseconds}us"
    elif isinstance(history_period, str) and re.fullmatch(
        r"(?:\d+(?:ns|us|ms|s|m|h|d|w))+", history_period
    ):
        period_ns = sum(
            int(value) * duration_units[unit]
            for value, unit in re.findall(
                r"(\d+)(ns|us|ms|s|m|h|d|w)", history_period
            )
        )
        label = history_period
    else:
        raise ValueError("history_period must be a positive fixed duration or timedelta")
    if not 0 < period_ns <= 2**63 - 1:
        raise ValueError("history_period must be positive and fit in Int64 nanoseconds")

    targets = (
        factors.select(pl.col("timestamp").cast(pl.Datetime("ns")))
        .lazy().with_row_index("_row_index")
    )
    qty = pl.col("qty").cast(pl.Float64)
    signed_qty = pl.when(pl.col("is_buyer_maker")).then(-qty).otherwise(qty)
    quantities = {}
    for group, condition in (
        ("rpi", pl.col("is_rpi")), ("non_rpi", ~pl.col("is_rpi"))
    ):
        quantities[f"_{group}_signed_qty"] = (
            pl.when(condition).then(signed_qty).otherwise(0.0)
        )
        quantities[f"_{group}_qty"] = (
            pl.when(condition).then(qty).otherwise(0.0)
        )
    trades = trade.select(
        pl.col("recv_ns").cast(pl.Datetime("ns")).alias("timestamp"),
        *(expr.alias(name) for name, expr in quantities.items()),
    ).drop_nulls("timestamp")
    target_events = (
        targets.select("timestamp").drop_nulls("timestamp").unique()
        .with_columns(*(pl.lit(0.0).alias(name) for name in quantities))
    )
    rpi_column = f"rpi_imbalance_{label}"
    non_rpi_column = f"non_rpi_imbalance_{label}"
    divergence_column = f"divergence_{label}"
    windows = (
        pl.concat([trades, target_events])
        .group_by("timestamp")
        .agg(*(pl.col(name).sum() for name in quantities))
        .sort("timestamp")
        .rolling("timestamp", period=f"{period_ns}ns", closed="right")
        .agg(*(pl.col(name).sum() for name in quantities))
        .select(
            "timestamp",
            *(
                pl.when(pl.col(f"_{group}_qty") > 0)
                .then(pl.col(f"_{group}_signed_qty") / pl.col(f"_{group}_qty"))
                .otherwise(0.0)
                .alias(column)
                for group, column in (
                    ("rpi", rpi_column), ("non_rpi", non_rpi_column)
                )
            ),
        )
        .with_columns(
            (pl.col(non_rpi_column) - pl.col(rpi_column)).alias(divergence_column)
        )
    )
    values = (
        targets.join(windows, on="timestamp", how="left")
        .sort("_row_index")
        .select(rpi_column, non_rpi_column, divergence_column)
        .collect()
    )
    return factors.with_columns(values.get_columns())
