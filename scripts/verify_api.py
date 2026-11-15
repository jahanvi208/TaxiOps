"""Smoke test the real containerized API and registered MLflow champion."""
from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8000"
PAYLOAD = {
    "trip_distance": 4.5,
    "passenger_count": 1,
    "pickup_zone_id": 161,
    "dropoff_zone_id": 237,
    "pickup_hour": 15,
    "pickup_weekday": 2,
    "pickup_month": 2,
}


def request(path: str, body: dict | None = None):
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Content-Type": "application/json"} if body is not None else {}
    req = urllib.request.Request(BASE + path, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as response:
        return response.status, response.read().decode()


def main() -> None:
    for _ in range(60):
        try:
            code, health = request("/health")
            if code == 200:
                break
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            time.sleep(2)
    else:
        sys.exit("FastAPI never became reachable; inspect docker compose logs")
    print("Health before prediction:", health)
    for _ in range(30):
        try:
            status, body = request("/predict", PAYLOAD)
            break
        except urllib.error.HTTPError as exc:
            if exc.code != 503:
                raise
            time.sleep(2)
    else:
        sys.exit("Registered champion prediction stayed unavailable (HTTP 503)")
    result = json.loads(body)
    if status != 200 or not isinstance(result.get("estimated_fare_usd"), (int, float)):
        sys.exit(f"Bad prediction response: {body}")
    if result.get("model_alias") != "champion":
        sys.exit(f"Wrong model alias: {body}")
    print("Live MLflow-registry prediction:", json.dumps(result, indent=2))
    code, metrics = request("/metrics")
    if code != 200 or "taxiops_http_requests_total" not in metrics:
        sys.exit("Prometheus metrics endpoint did not return expected counter")
    try:
        request("/predict", {**PAYLOAD, "pickup_hour": 42})
    except urllib.error.HTTPError as exc:
        if exc.code != 422:
            raise
    else:
        sys.exit("Invalid prediction input was not rejected with 422")
    print("PASS: /health, /predict (real registry model), /metrics, invalid-input 422")


if __name__ == "__main__":
    main()
