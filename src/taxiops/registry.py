"""Centralized MLflow registry loading and alias promotion."""
from __future__ import annotations

import os


def set_tracking() -> None:
    import mlflow

    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not uri:
        hostport = os.environ.get("MLFLOW_TRACKING_HOSTPORT")
        uri = f"http://{hostport}" if hostport else "http://localhost:5000"
    mlflow.set_tracking_uri(uri)


def load_champion(model_name: str, alias: str = "champion"):
    import mlflow

    set_tracking()
    return mlflow.pyfunc.load_model(f"models:/{model_name}@{alias}")


def champion_version(model_name: str, alias: str = "champion"):
    from mlflow.tracking import MlflowClient

    set_tracking()
    client = MlflowClient()
    try:
        return client.get_model_version_by_alias(model_name, alias)
    except Exception as exc:
        # Only an actual missing alias is a bootstrap condition; other failures must fail.
        from mlflow.exceptions import MlflowException

        if isinstance(exc, MlflowException) and (
            getattr(exc, "error_code", None) == "RESOURCE_DOES_NOT_EXIST"
            or "not found" in str(exc).lower()
            or "does not exist" in str(exc).lower()
        ):
            return None
        raise
