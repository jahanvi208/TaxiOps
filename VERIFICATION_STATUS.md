# TaxiOps validation status

## Completed in the artifact-building environment (2026-11-15)

- Reviewed the source files from the earlier TaxiOps ZIP.
- `PYTHONPATH=src pytest -q`: **39 passed**.
- Included an actual XGBoost fit/predict integration test using generated, deterministic synthetic tabular data. The test demonstrates that preprocessing and estimation execute, but it is **not** an NYC fare accuracy measurement.
- Python syntax compilation succeeded for the source, scripts, and tests.
- Parsed the GitHub Actions workflow YAML, confirming the `quality`, `model-regression`, `cloud-verification`, and `publish` job definitions.

## Not yet verified

- Official January and February 2025 NYC taxi Parquet downloads and DVC tracking (this execution environment cannot access external binary downloads).
- Running Docker Compose, the MLflow server and registry, API inference against the registered model, GHCR publication, or a Render deployment (Docker is unavailable in this environment).
- Fifteen actual MLflow tuning runs; baseline/model RMSE on real TLC trips; live Locust p50/p95/RPS; actual container image size; a public API URL.

## How to finish without local Docker

Push this source repository to GitHub `main`. Its Actions workflow runs the real data ingestion, Docker build/compose, 15-run training, registry/API smoke tests, Locust load test, and evidence collection on a GitHub-hosted Linux VM. A successful run uploads `taxiops-cloud-verification`, the measured evidence bundle. The hosted registry is an ephemeral test instance, not a permanent cloud deployment. A persistent MLflow service and live public API still require cloud provisioning.

No unmeasured performance numbers are asserted.
