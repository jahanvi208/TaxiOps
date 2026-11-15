from __future__ import annotations

import pytest
from pydantic import ValidationError

from taxiops.api.app import TripRequest


def test_valid_schema(valid_request):
    assert TripRequest(**valid_request).pickup_zone_id == 161


@pytest.mark.parametrize("key,value", [
    ("trip_distance", -1.0),
    ("trip_distance", 101.0),
    ("passenger_count", 0),
    ("passenger_count", 9),
    ("pickup_hour", 24),
    ("pickup_weekday", 7),
    ("pickup_month", 13),
    ("pickup_zone_id", 0),
    ("dropoff_zone_id", 266),
])
def test_out_of_range_inputs_rejected(valid_request, key, value):
    with pytest.raises(ValidationError):
        TripRequest(**{**valid_request, key: value})


def test_missing_required_value_rejected(valid_request):
    del valid_request["pickup_hour"]
    with pytest.raises(ValidationError):
        TripRequest(**valid_request)


def test_extra_field_rejected(valid_request):
    with pytest.raises(ValidationError):
        TripRequest(**{**valid_request, "total_amount": 100.0})


def test_string_number_rejected(valid_request):
    with pytest.raises(ValidationError):
        TripRequest(**{**valid_request, "passenger_count": "2"})
