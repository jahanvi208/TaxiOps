"""Shared feature contract for data processing, training, and inference."""
from __future__ import annotations

import os

MODEL_NAME = os.getenv("TAXIOPS_MODEL_NAME", "TaxiOpsFare")
MODEL_ALIAS = os.getenv("TAXIOPS_MODEL_ALIAS", "champion")
NUMERIC_FEATURES = ("trip_distance", "passenger_count")
CATEGORICAL_FEATURES = (
    "pickup_zone_id",
    "dropoff_zone_id",
    "pickup_hour",
    "pickup_weekday",
    "pickup_month",
)
FEATURE_COLUMNS = list(NUMERIC_FEATURES + CATEGORICAL_FEATURES)
TARGET = "fare_amount"
SOURCE_COLUMNS = [
    "tpep_pickup_datetime",
    "trip_distance",
    "passenger_count",
    "PULocationID",
    "DOLocationID",
    "fare_amount",
]
