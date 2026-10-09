"""Order flow imbalance from changes in best bid/ask quotes."""

from datetime import timedelta
import re

import polars as pl


def generate_ofi_factor(
    factors: pl.DataFrame,
    bbo: pl.LazyFrame,
    trade: pl.LazyFrame,
    depth: pl.LazyFrame,
    period: str | timedelta,
) -> pl.DataFrame:
    """Append ``ofi_{period}``, summing BBO contributions in (t-period, t].

    Bid contribution: current bid_qty when bid_px rises, quantity difference
    when unchanged, minus previous bid_qty when it falls. Ask contribution:
    minus previous ask_qty when ask_px rises, quantity difference when
    unchanged, current ask_qty when it falls. OFI is bid minus ask.

    Sort BBO by recv_ns, retaining input order for equal timestamps. Compute
    each update against the preceding BBO, including quotes outside the
    window; the first BBO initializes the state and has no contribution.
    All updates at equal timestamps are included. History is complete once
    t-period >= the first BBO timestamp. Incomplete history, empty BBO and
    null targets yield null; complete windows with no updates yield 0.0.

    Inputs must be for one symbol with Int64 ns timestamps and non-null
    bid_px, ask_px, bid_qty and ask_qty. Output uses supplied quantity units,
    without normalization. Period accepts positive fixed duration strings
    (ns/us/ms/s/m/h/d/w) or timedelta. Preserve input rows/order/columns and
    replace an existing output column. Trade and depth are not evaluated.
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

    factor_column = f"ofi_{label}"
    targets = (
        factors.select(pl.col("timestamp").cast(pl.Datetime("ns")))
        .lazy().with_row_index("_row_index")
    )
    quotes = (
        bbo.select(
            pl.col("recv_ns").cast(pl.Datetime("ns")).alias("timestamp"),
            "bid_px", "ask_px",
            pl.col("bid_qty").cast(pl.Float64),
            pl.col("ask_qty").cast(pl.Float64),
        )
        .drop_nulls("timestamp")
        .sort("timestamp", maintain_order=True)
    )
    bid_delta = (
        pl.when(pl.col("bid_px") > pl.col("bid_px").shift(1))
        .then(pl.col("bid_qty"))
        .when(pl.col("bid_px") == pl.col("bid_px").shift(1))
        .then(pl.col("bid_qty") - pl.col("bid_qty").shift(1))
        .otherwise(-pl.col("bid_qty").shift(1))
    )
    ask_delta = (
        pl.when(pl.col("ask_px") > pl.col("ask_px").shift(1))
        .then(-pl.col("ask_qty").shift(1))
        .when(pl.col("ask_px") == pl.col("ask_px").shift(1))
        .then(pl.col("ask_qty") - pl.col("ask_qty").shift(1))
        .otherwise(pl.col("ask_qty"))
    )
    updates = quotes.select(
        "timestamp",
        pl.when(pl.col("bid_px").shift(1).is_null())
        .then(0.0).otherwise(bid_delta - ask_delta).alias("_ofi"),
    )
    target_events = (
        targets.select("timestamp").drop_nulls("timestamp").unique()
        .with_columns(pl.lit(0.0).alias("_ofi"))
    )
    windows = (
        pl.concat([updates, target_events])
        .group_by("timestamp").agg(pl.col("_ofi").sum())
        .sort("timestamp")
        .rolling("timestamp", period=f"{period_ns}ns", closed="right")
        .agg(pl.col("_ofi").sum())
    )
    values = (
        targets.join(windows, on="timestamp", how="left")
        .join(quotes.select(pl.col("timestamp").min().alias("_bbo_start")), how="cross")
        .sort("_row_index")
        .select(
            pl.when(
                pl.col("timestamp").is_null()
                | pl.col("_bbo_start").is_null()
                | ((pl.col("timestamp") - pl.col("_bbo_start"))
                   .dt.total_nanoseconds() < period_ns)
            )
            .then(None).otherwise(pl.col("_ofi")).alias(factor_column)
        )
        .collect()
    )
    return factors.with_columns(values.get_columns())
