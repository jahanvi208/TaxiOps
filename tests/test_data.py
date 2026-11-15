from __future__ import annotations

import pandas as pd
import pytest

from taxiops.config import FEATURE_COLUMNS
from taxiops.data import clean_trips, validate_source_columns


def test_missing_source_columns_raise():
    with pytest.raises(ValueError, match="Missing NYC TLC columns"):
        validate_source_columns(["trip_distance", "fare_amount"])


def test_keep_only_valid_requested_month(sample_raw):
    cleaned = clean_trips(sample_raw, "2025-01")
    assert len(cleaned) == 1
    assert cleaned.iloc[0]["fare_amount"] == pytest.approx(19.5)


def test_derived_pickup_features(sample_raw):
    result = clean_trips(sample_raw, "2025-01")
    assert result.iloc[0]["pickup_hour"] == 14
    assert result.iloc[0]["pickup_weekday"] == 2
    assert result.iloc[0]["pickup_month"] == 1
    assert all(col in result.columns for col in FEATURE_COLUMNS)


def test_fare_target_does_not_leak_into_features():
    assert "fare_amount" not in FEATURE_COLUMNS
    assert "total_amount" not in FEATURE_COLUMNS
    assert "tip_amount" not in FEATURE_COLUMNS


def test_fare_outliers_are_rejected(sample_raw):
    sample_raw.loc[0, "fare_amount"] = 900
    assert clean_trips(sample_raw, "2025-01").empty


def test_invalid_pickup_time_is_rejected(sample_raw):
    sample_raw.loc[0, "tpep_pickup_datetime"] = "bad-date"
    assert clean_trips(sample_raw, "2025-01").empty


def test_non_numeric_distance_rejected(sample_raw):
    sample_raw["trip_distance"] = sample_raw["trip_distance"].astype(object)
    sample_raw.loc[0, "trip_distance"] = "bad"
    assert clean_trips(sample_raw, "2025-01").empty


def test_clean_does_not_mutate_source(sample_raw):
    before = sample_raw.copy(deep=True)
    clean_trips(sample_raw, "2025-01")
    pd.testing.assert_frame_equal(before, sample_raw)


def test_fractional_passenger_count_rejected(sample_raw):
    sample_raw["passenger_count"] = sample_raw["passenger_count"].astype(float)
    sample_raw.loc[0, "passenger_count"] = 1.5
    assert clean_trips(sample_raw, "2025-01").empty


def test_fractional_zone_id_rejected(sample_raw):
    sample_raw["PULocationID"] = sample_raw["PULocationID"].astype(float)
    sample_raw.loc[0, "PULocationID"] = 161.5
    assert clean_trips(sample_raw, "2025-01").empty
