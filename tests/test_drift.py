import pandas as pd
import pytest

from taxiops.drift import measure_drift


def test_identical_distribution_has_no_drift():
    frame = pd.DataFrame(
        {"trip_distance": [1.0, 2.0, 3.0, 4.0],
         "fare_amount": [8.0, 10.0, 15.0, 18.0],
         "pickup_zone_id": [161, 161, 237, 237],
         "pickup_hour": [12, 13, 14, 15]}
    )
    result = measure_drift(frame, frame)
    assert result["numeric"]["trip_distance"]["ks_statistic"] == pytest.approx(0)
    assert result["categorical"]["pickup_zone_id"]["jensen_shannon_distance"] == pytest.approx(0)


def test_disjoint_distributions_are_detected():
    a = pd.DataFrame({"trip_distance": [1.0, 1.0], "fare_amount": [8.0, 8.0],
                      "pickup_zone_id": [100, 100], "pickup_hour": [7, 7]})
    b = pd.DataFrame({"trip_distance": [10.0, 10.0], "fare_amount": [50.0, 50.0],
                      "pickup_zone_id": [200, 200], "pickup_hour": [19, 19]})
    output = measure_drift(a, b)
    assert output["numeric"]["trip_distance"]["ks_statistic"] == pytest.approx(1)
    assert output["categorical"]["pickup_zone_id"]["jensen_shannon_distance"] == pytest.approx(1)
