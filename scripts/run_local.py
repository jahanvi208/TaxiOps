"""Build and start local compose, train 15 real-data models, then smoke-test /predict."""
from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    subprocess.run(["docker", "compose", "up", "-d", "--build"], cwd=ROOT, check=True)
    for _ in range(120):
        try:
            with urllib.request.urlopen("http://127.0.0.1:5000/", timeout=2):
                break
        except (urllib.error.URLError, TimeoutError):
            time.sleep(2)
    else:
        sys.exit("MLflow did not become reachable at http://127.0.0.1:5000")
    subprocess.run([sys.executable, "-m", "taxiops.train", "--trials", "15"],
                   cwd=ROOT, check=True)
    payload = {
        "trip_distance": 4.5, "passenger_count": 1, "pickup_zone_id": 161,
        "dropoff_zone_id": 237, "pickup_hour": 15,
        "pickup_weekday": 2, "pickup_month": 2,
    }
    request = urllib.request.Request(
        "http://127.0.0.1:8000/predict", json.dumps(payload).encode(),
        {"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        print("Smoke-test prediction:", response.read().decode())


if __name__ == "__main__":
    main()
