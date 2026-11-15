"""Parquet schema checks, deterministic cleaning, and bounded-memory sampling."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from taxiops.config import FEATURE_COLUMNS, SOURCE_COLUMNS, TARGET


def validate_source_columns(columns: list[str] | set[str]) -> None:
    missing = set(SOURCE_COLUMNS) - set(columns)
    if missing:
        raise ValueError(f"Missing NYC TLC columns: {sorted(missing)}")


def clean_trips(raw: pd.DataFrame, month: str) -> pd.DataFrame:
    """Keep only plausible trips from the requested month; never use outcome leakage."""
    validate_source_columns(set(raw.columns))
    data = raw[SOURCE_COLUMNS].copy()
    pickup_time = pd.to_datetime(data["tpep_pickup_datetime"], errors="coerce", format="mixed")
    data["pickup_hour"] = pickup_time.dt.hour
    data["pickup_weekday"] = pickup_time.dt.dayofweek
    data["pickup_month"] = pickup_time.dt.month
    data["pickup_zone_id"] = pd.to_numeric(data["PULocationID"], errors="coerce")
    data["dropoff_zone_id"] = pd.to_numeric(data["DOLocationID"], errors="coerce")
    for name in ("trip_distance", "passenger_count", TARGET):
        data[name] = pd.to_numeric(data[name], errors="coerce")
    # Drop impossible/atypical charges rather than cap the target to a fabricated value.
    valid = (
        pickup_time.dt.strftime("%Y-%m").eq(month)
        & data["trip_distance"].between(0.1, 100)
        & data[TARGET].between(1, 250)
        & data["passenger_count"].between(1, 8)
        & data["passenger_count"].eq(np.floor(data["passenger_count"]))
        & data["pickup_zone_id"].between(1, 265)
        & data["pickup_zone_id"].eq(np.floor(data["pickup_zone_id"]))
        & data["dropoff_zone_id"].between(1, 265)
        & data["dropoff_zone_id"].eq(np.floor(data["dropoff_zone_id"]))
    )
    selected = data.loc[valid, FEATURE_COLUMNS + [TARGET]].copy()
    if not selected.empty:
        integer_cols = [
            "passenger_count", "pickup_zone_id", "dropoff_zone_id",
            "pickup_hour", "pickup_weekday", "pickup_month",
        ]
        selected[integer_cols] = selected[integer_cols].astype("int16")
        selected[["trip_distance", TARGET]] = selected[["trip_distance", TARGET]].astype(
            "float32"
        )
    return selected


def load_month(path: str | Path, month: str, max_rows: int = 100_000, seed: int = 42):
    """Uniform random-priority sample of valid trips over all Parquet row groups.

    Unlike taking the first N records, selecting the smallest random priorities
    provides each valid row equal probability of appearing in the final sample.
    """
    import pyarrow.parquet as pq

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing data: {path}. Run scripts/download_data.py first.")
    if max_rows <= 0:
        raise ValueError("max_rows must be a positive integer")
    parquet = pq.ParquetFile(path)
    validate_source_columns(set(parquet.schema.names))
    rng = np.random.default_rng(seed)
    sampled = pd.DataFrame()
    raw_rows = valid_rows = 0
    for record_batch in parquet.iter_batches(columns=SOURCE_COLUMNS, batch_size=50_000):
        raw = record_batch.to_pandas()
        raw_rows += len(raw)
        cleaned = clean_trips(raw, month)
        valid_rows += len(cleaned)
        if cleaned.empty:
            continue
        cleaned["_priority"] = rng.random(len(cleaned))
        sampled = pd.concat([sampled, cleaned], ignore_index=True)
        if len(sampled) > max_rows:
            sampled = sampled.nsmallest(max_rows, "_priority")
    if sampled.empty:
        raise ValueError(f"No valid trips found in {path} for month {month}")
    sampled = sampled.drop(columns="_priority").reset_index(drop=True)
    report = {
        "source": str(path),
        "month": month,
        "raw_rows": raw_rows,
        "valid_rows": valid_rows,
        "sampled_rows": len(sampled),
        "rejected_rows": raw_rows - valid_rows,
        "seed": seed,
    }
    return sampled, report
