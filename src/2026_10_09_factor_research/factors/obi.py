"""Best-bid/ask order book imbalance at requested timestamps."""

import polars as pl


def generate_obi_factor(
    factors: pl.DataFrame,
    bbo: pl.LazyFrame,
    trade: pl.LazyFrame,
    depth: pl.LazyFrame,
) -> pl.DataFrame:
    """Return factors with an ``obi`` Float64 column appended (or replaced).

    Inputs must contain data for a single symbol. ``factors.timestamp`` and
    the BBO time column must have matching types/units; the recorder uses
    Int64 nanoseconds. Use receive time to avoid using quotes before they
    were received.

    Each target uses the last BBO at or before its timestamp. Equal BBO
    timestamps use the last input row. OBI is (bid_qty - ask_qty) /
    (bid_qty + ask_qty); missing history or zero total quantity yields null.
    Original rows, their order, and other columns are preserved. The input
    DataFrame is not mutated. Trade and depth are accepted for the common
    factor interface and are not evaluated.
    """
    bbo_timestamp_column = "recv_ns"
    bid_qty = pl.col("bid_qty").cast(pl.Float64)
    ask_qty = pl.col("ask_qty").cast(pl.Float64)
    total_qty = bid_qty + ask_qty
    quotes = (
        bbo.select(
            pl.col(bbo_timestamp_column).alias("timestamp"),
            pl.when(total_qty != 0)
            .then((bid_qty - ask_qty) / total_qty)
            .otherwise(None)
            .alias("obi"),
        )
        .drop_nulls("timestamp")
        .sort("timestamp", maintain_order=True)
    )
    obi = (
        factors.select("timestamp")
        .lazy()
        .with_row_index("_row_index")
        .sort("timestamp")
        .join_asof(quotes, on="timestamp", strategy="backward")
        .sort("_row_index")
        .select("obi")
        .collect()
        .get_column("obi")
    )
    return factors.with_columns(obi)


def generate_multi_obi_factor(
    factors: pl.DataFrame,
    bbo: pl.LazyFrame,
    trade: pl.LazyFrame,
    depth: pl.LazyFrame,
    levels: int,
) -> pl.DataFrame:
    """Append ``obi_{levels}`` using the first N levels of depth quantities.

    Depth must contain ``recv_ns``, ``bid_qty_1`` through ``bid_qty_N``,
    and the corresponding ``ask_qty_*`` columns for one symbol. Its time
    type/unit must match ``factors.timestamp`` (Int64 nanoseconds for the
    recorder). Levels must be a positive integer.

    OBI is (sum(bid_qty) - sum(ask_qty)) / (sum(bid_qty) + sum(ask_qty)).
    Use the last depth row received at or before each target timestamp;
    ties use the last input row. Missing history, missing quantities, or
    zero total quantity yields null. Preserve input rows, order, and other
    columns without mutation; replace the output column if it exists.
    BBO and trade are accepted for the common interface and not evaluated.
    """
    if isinstance(levels, bool) or not isinstance(levels, int) or levels < 1:
        raise ValueError("levels must be a positive integer")

    depth_timestamp_column = "recv_ns"
    factor_column = f"obi_{levels}"
    bid_qty = pl.sum_horizontal(
        [pl.col(f"bid_qty_{level}").cast(pl.Float64)
         for level in range(1, levels + 1)],
        ignore_nulls=False,
    )
    ask_qty = pl.sum_horizontal(
        [pl.col(f"ask_qty_{level}").cast(pl.Float64)
         for level in range(1, levels + 1)],
        ignore_nulls=False,
    )
    total_qty = bid_qty + ask_qty
    quotes = (
        depth.select(
            pl.col(depth_timestamp_column).alias("timestamp"),
            pl.when(total_qty != 0)
            .then((bid_qty - ask_qty) / total_qty)
            .otherwise(None)
            .alias(factor_column),
        )
        .drop_nulls("timestamp")
        .sort("timestamp", maintain_order=True)
    )
    obi = (
        factors.select("timestamp")
        .lazy()
        .with_row_index("_row_index")
        .sort("timestamp")
        .join_asof(quotes, on="timestamp", strategy="backward")
        .sort("_row_index")
        .select(factor_column)
        .collect()
        .get_column(factor_column)
    )
    return factors.with_columns(obi)
