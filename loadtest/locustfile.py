"""Load test against a trained, live API; HTML report includes latency percentiles and RPS."""
from __future__ import annotations

from locust import HttpUser, between, task


class TaxiClient(HttpUser):
    wait_time = between(0.05, 0.3)

    @task(10)
    def prediction(self):
        self.client.post(
            "/predict",
            json={
                "trip_distance": 4.5, "passenger_count": 1,
                "pickup_zone_id": 161, "dropoff_zone_id": 237,
                "pickup_hour": 15, "pickup_weekday": 2, "pickup_month": 2,
            },
        )

    @task(1)
    def health(self):
        self.client.get("/health")
