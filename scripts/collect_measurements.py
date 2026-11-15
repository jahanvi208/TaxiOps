"""Collect only observed MLOps benchmarks; missing evidence remains null."""
from __future__ import annotations

import csv
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"


def main():
    report_file = OUT / "training_report.json"
    train = json.loads(report_file.read_text()) if report_file.is_file() else {}
    result = {
        "mlflow_tuning_runs": train.get("trials"),
        "mlflow_selected_model_run": 1 if train.get("promoted") else None,
        "training_rows": train.get("train_data", {}).get("sampled_rows"),
        "validation_rows": train.get("validation_data", {}).get("sampled_rows"),
        "naive_baseline_validation_rmse_usd": train.get("baseline_validation_rmse"),
        "best_model_validation_rmse_usd": train.get("best_validation_rmse"),
        "test_count": None,
        "docker_image_bytes": None,
        "predict_p50_ms": None,
        "predict_p95_ms": None,
        "predict_requests_per_second": None,
        "predict_failed_requests": None,
    }
    tests = subprocess.run(
        ["python", "-m", "pytest", "--collect-only", "-q"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    if tests.returncode == 0:
        match = re.search(r"(\d+) tests? collected", tests.stdout)
        if match:
            result["test_count"] = int(match.group(1))
    image = subprocess.run(
        ["docker", "image", "inspect", "taxiops:dev", "--format", "{{.Size}}"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    ) if executable_available("docker") else None
    if image is not None and image.returncode == 0 and image.stdout.strip().isdigit():
        result["docker_image_bytes"] = int(image.stdout.strip())
    locust = OUT / "locust_stats.csv"
    if locust.is_file():
        with locust.open(newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                if row.get("Type") == "POST" and row.get("Name") == "/predict":
                    for key, metric in (
                        ("50%", "predict_p50_ms"), ("95%", "predict_p95_ms"),
                        ("Requests/s", "predict_requests_per_second"),
                    ):
                        if row.get(key):
                            result[metric] = float(row[key])
                    if row.get("Failure Count"):
                        result["predict_failed_requests"] = int(row["Failure Count"])
    destination = OUT / "benchmark_evidence.json"
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved: {destination}")


def executable_available(name: str) -> bool:
    from shutil import which

    return which(name) is not None


if __name__ == "__main__":
    main()
