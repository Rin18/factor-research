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
