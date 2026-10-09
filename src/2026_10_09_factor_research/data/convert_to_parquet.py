"""Merge recorder segments into one Parquet per stream/symbol.

Run from the project root:
    uv run python src/2026_10_09_factor_research/data/convert_to_parquet.py

Configuration is read from the adjacent config.toml [convert_to_parquet]
section. Relative input/output paths are resolved against that file.

Prices/quantities remain fixed-point int64; divide by 10**scale (schema
metadata) for decimals. Timestamps remain int64 nanoseconds. Depth stores
only valid levels as list<struct<px: int64, qty: int64>>. Source segments
are ordered by their filename's hour/session; records are not sorted or
deduplicated inside segments. Only finalized .bin.zst files are included.
"""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
import json
from pathlib import Path
import struct
import tempfile
import tomllib

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import zstandard as zstd


CONFIG_PATH = Path(__file__).resolve().with_name("config.toml")
COMMON = [("recv_ns", "<i8"), ("event_ns", "<i8"), ("transact_ns", "<i8")]
DTYPES = {
    "bbo": np.dtype(COMMON + [
        ("update_id", "<u8"), ("bid_px", "<i8"), ("bid_qty", "<i8"),
        ("ask_px", "<i8"), ("ask_qty", "<i8"),
    ]),
    "trade": np.dtype(COMMON + [
        ("trade_id", "<u8"), ("px", "<i8"), ("qty", "<i8"),
        ("flags", "u1"), ("_pad", "u1", 7),
    ]),
    "depth": np.dtype(COMMON + [
        ("update_id", "<u8"), ("n_bids", "u1"), ("n_asks", "u1"),
        ("_pad", "u1", 6), ("bids", "<i8", (100, 2)),
        ("asks", "<i8", (100, 2)),
    ]),
}
LEVEL_TYPE = pa.list_(pa.struct([("px", pa.int64()), ("qty", pa.int64())]))


@dataclass(frozen=True)
class ConversionConfig:
    input: Path
    output: Path
    symbols: list[str] | None
    streams: list[str]
    batch_rows: int
    overwrite: bool


def load_config(path: Path = CONFIG_PATH) -> ConversionConfig:
    with path.open("rb") as source:
        config = tomllib.load(source).get("convert_to_parquet")
    if not isinstance(config, dict):
        raise ValueError(f"Missing [convert_to_parquet] section in {path}")
    required = {"input", "output", "symbols", "streams", "batch_rows", "overwrite"}
    if config.keys() != required:
        raise ValueError(
            f"Invalid configuration keys: missing={sorted(required - config.keys())}, "
            f"unknown={sorted(config.keys() - required)}"
        )
    for key in ("input", "output"):
        if not isinstance(config[key], str) or not config[key].strip():
            raise ValueError(f"{key} must be a nonempty path string")
    for key in ("symbols", "streams"):
        if not isinstance(config[key], list) or not all(
            isinstance(value, str) and value.strip() for value in config[key]
        ):
            raise ValueError(f"{key} must be an array of nonempty strings")
    if not config["streams"] or any(stream not in DTYPES for stream in config["streams"]):
        raise ValueError("streams must contain bbo, trade and/or depth")
    if type(config["batch_rows"]) is not int or not 1 <= config["batch_rows"] <= 1_000_000:
        raise ValueError("batch_rows must be an integer between 1 and 1,000,000")
    if not isinstance(config["overwrite"], bool):
        raise ValueError("overwrite must be true or false")
    return ConversionConfig(
        input=(path.resolve().parent / Path(config["input"]).expanduser()).resolve(),
        output=(path.resolve().parent / Path(config["output"]).expanduser()).resolve(),
        symbols=config["symbols"] or None,
        streams=list(dict.fromkeys(config["streams"])),
        batch_rows=config["batch_rows"],
        overwrite=config["overwrite"],
    )


def read_exact(reader, size: int) -> bytes:
    """Allow short reads while rejecting truncated headers/records."""
    chunks = []
    remaining = size
    while remaining:
        chunk = reader.read(remaining)
        if not chunk:
            raise ValueError(f"Truncated recording: missing {remaining} bytes")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


@contextmanager
def open_recording(path: Path):
    with path.open("rb") as source:
        with zstd.ZstdDecompressor().stream_reader(source, read_across_frames=True) as reader:
            header_size = struct.unpack("<I", read_exact(reader, 4))[0]
            if not 0 < header_size <= 1 << 20:
                raise ValueError(f"Invalid header length in {path}: {header_size}")
            header = json.loads(read_exact(reader, header_size))
            if header.get("version") != 1 or header.get("stream") not in DTYPES:
                raise ValueError(f"Unsupported recording header in {path}: {header}")
            if header.get("record_size") != DTYPES[header["stream"]].itemsize:
                raise ValueError(f"Record size mismatch in {path}")
            if not isinstance(header.get("scale"), int) or header["scale"] < 0:
                raise ValueError(f"Invalid fixed-point scale in {path}")
            yield header, reader


def record_batches(reader, dtype: np.dtype, batch_rows: int):
    target = dtype.itemsize * batch_rows
    while True:
        chunks = []
        remaining = target
        while remaining:
            chunk = reader.read(remaining)
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        if not raw:
            return
        if len(raw) % dtype.itemsize:
            raise ValueError("Finalized recording ends with an incomplete record")
        yield np.frombuffer(raw, dtype=dtype)


def schema_for(stream: str, header: dict) -> pa.Schema:
    fields = []
    for name in DTYPES[stream].names:
        if name in ("_pad", "n_bids", "n_asks"):
            continue
        arrow_type = LEVEL_TYPE if name in ("bids", "asks") else pa.from_numpy_dtype(
            DTYPES[stream].fields[name][0]
        )
        fields.append(pa.field(name, arrow_type, nullable=False))
    # hour/session vary across segments and are recorded in source_headers instead.
    stable_header = {k: v for k, v in header.items() if k not in ("hour", "session")}
    metadata = {k: json.dumps(v).encode() for k, v in stable_header.items()}
    metadata["time_unit"] = b"ns"
    metadata["price_quantity_encoding"] = b"fixed_point_int64"
    if stream == "trade":
        metadata["flag_buyer_maker"] = b"1"
        metadata["flag_rpi"] = b"2"
    return pa.schema(fields, metadata=metadata)


def depth_array(records: np.ndarray, side: str) -> pa.ListArray:
    counts = records[f"n_{side}"].astype(np.int32)
    if np.any(counts > 100):
        raise ValueError(f"Invalid {side} count: exceeds 100 levels")
    offsets = np.empty(len(records) + 1, dtype=np.int32)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])
    valid = np.arange(100)[None, :] < counts[:, None]
    levels = records[side][valid]
    values = pa.StructArray.from_arrays(
        [pa.array(levels[:, 0]), pa.array(levels[:, 1])], names=["px", "qty"]
    )
    return pa.ListArray.from_arrays(offsets, values)


def to_table(records: np.ndarray, schema: pa.Schema) -> pa.Table:
    arrays = [
        depth_array(records, field.name) if field.name in ("bids", "asks")
        else pa.array(records[field.name], type=field.type)
        for field in schema
    ]
    return pa.Table.from_arrays(arrays, schema=schema)


def discover(input_dir: Path, symbols: list[str] | None, streams: list[str]):
    groups = defaultdict(list)
    for stream in streams:
        for path in sorted((input_dir / stream).glob("*/*/*.bin.zst")):
            symbol = path.parent.parent.name
            if symbols is None or symbol in symbols:
                groups[stream, symbol].append(path)
    for paths in groups.values():
        paths.sort(key=lambda path: (path.name, str(path)))
    return groups


def convert_group(
    stream: str, symbol: str, paths: list[Path], output: Path,
    batch_rows: int = 8192, overwrite: bool = False,
) -> int | None:
    if output.exists() and not overwrite:
        print(f"Skip {output}: already exists (set overwrite = true in config.toml to rebuild)", flush=True)
        return None
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=output.parent, prefix=f".{output.stem}.", suffix=".tmp", delete=False,
    ) as temporary:
        temp_path = Path(temporary.name)
    rows = 0
    schema = None
    writer = None
    headers = []
    try:
        for index, path in enumerate(paths, start=1):
            with open_recording(path) as (header, reader):
                if header.get("stream") != stream or header.get("symbol") != symbol:
                    raise ValueError(f"Stream/symbol does not match directory: {path}")
                candidate = schema_for(stream, header)
                if schema is None:
                    schema = candidate
                    writer = pq.ParquetWriter(
                        temp_path, schema, compression="zstd", use_dictionary=False,
                    )
                elif not schema.equals(candidate, check_metadata=True):
                    raise ValueError(f"Incompatible header (including scale/fields): {path}")
                headers.append({"file": str(path), "header": header})
                for records in record_batches(reader, DTYPES[stream], batch_rows):
                    table = to_table(records, schema)
                    writer.write_table(table, row_group_size=batch_rows)
                    rows += len(records)
            if index == 1 or index % 10 == 0 or index == len(paths):
                print(f"{stream}/{symbol}: {index}/{len(paths)} files, {rows:,} rows", flush=True)
        if writer is None:
            raise ValueError("No source files provided")
        writer.add_key_value_metadata({"source_headers": json.dumps(headers)})
        writer.close()
        writer = None
        if pq.ParquetFile(temp_path).metadata.num_rows != rows:
            raise ValueError(f"Output row count mismatch: {output}")
        temp_path.replace(output)
    finally:
        try:
            if writer is not None:
                writer.close()
        finally:
            temp_path.unlink(missing_ok=True)
    print(f"Written {output}: {rows:,} rows", flush=True)
    return rows


def main() -> None:
    config = load_config()
    groups = discover(config.input, config.symbols, config.streams)
    if not groups:
        raise ValueError(f"No matching finalized .bin.zst recordings under {config.input}")
    for (stream, symbol), paths in sorted(groups.items()):
        try:
            convert_group(
                stream, symbol, paths, config.output / stream / f"{symbol}.parquet",
                config.batch_rows, config.overwrite,
            )
        except Exception as error:
            raise RuntimeError(f"Failed converting {stream}/{symbol}: {error}") from error


if __name__ == "__main__":
    main()
