from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def sample_raw():
    return pd.DataFrame(
        {
            "tpep_pickup_datetime": ["2025-01-15 14:30:00", "2025-02-01 12:00:00",
                                      "2025-01-15 14:30:00"],
            "trip_distance": [3.2, 2.0, -4.0],
            "passenger_count": [1, 2, 1],
            "PULocationID": [161, 161, 161],
            "DOLocationID": [237, 237, 237],
            "fare_amount": [19.5, 15.5, 20.0],
        }
    )


@pytest.fixture
def valid_request():
    return {
        "trip_distance": 4.5,
        "passenger_count": 1,
        "pickup_zone_id": 161,
        "dropoff_zone_id": 237,
        "pickup_hour": 15,
        "pickup_weekday": 2,
        "pickup_month": 2,
    }
