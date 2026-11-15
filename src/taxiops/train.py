"""Month-separated experiments with ≥15 tracked trials and a registry gate."""
from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.model_selection import ParameterSampler

from taxiops.config import FEATURE_COLUMNS, MODEL_ALIAS, MODEL_NAME, TARGET
from taxiops.data import load_month
from taxiops.drift import measure_drift
from taxiops.metrics import regression_passes, rmse
from taxiops.model import make_pipeline
from taxiops.registry import champion_version, load_champion, set_tracking

SEARCH_SPACE = {
    "n_estimators": [100, 160, 220],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.03, 0.06, 0.1],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
    "reg_lambda": [1, 5, 10],
    "min_child_weight": [1, 3],
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate monthly taxi-fare models")
    parser.add_argument("--train-month", default="2025-01")
    parser.add_argument("--val-month", default="2025-02")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--max-rows", type=int, default=100_000)
    parser.add_argument("--trials", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report", type=Path, default=Path("outputs/training_report.json"))
    parser.add_argument("--no-promote", action="store_true")
    parser.add_argument("--require-champion", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    if args.train_month >= args.val_month:
        raise ValueError("The validation month must occur strictly after the training month")
    if args.trials < 1:
        raise ValueError("At least one trial is required")
    train_path = args.data_dir / f"yellow_tripdata_{args.train_month}.parquet"
    val_path = args.data_dir / f"yellow_tripdata_{args.val_month}.parquet"
    train, train_stats = load_month(train_path, args.train_month, args.max_rows, args.seed)
    val, val_stats = load_month(val_path, args.val_month, args.max_rows, args.seed + 1)
    x_train, y_train = train[FEATURE_COLUMNS], train[TARGET]
    x_val, y_val = val[FEATURE_COLUMNS], val[TARGET]

    # Naive baseline uses only the training mean; validation targets never influence it.
    baseline = rmse(y_val, np.full(len(y_val), float(y_train.mean())))
    drift_report = measure_drift(train, val)
    set_tracking()
    mlflow.set_experiment("taxiops-monthly-fare")
    candidates = list(ParameterSampler(SEARCH_SPACE, n_iter=args.trials, random_state=args.seed))
    best_rmse, best_params, best_model, best_run = float("inf"), None, None, None
    for trial, params in enumerate(candidates, start=1):
        pipeline = make_pipeline(params, seed=args.seed)
        with mlflow.start_run(run_name=f"tune-{trial:02d}") as run:
            mlflow.log_params(
                {**params, "train_month": args.train_month, "val_month": args.val_month,
                 "seed": args.seed, "max_rows": args.max_rows}
            )
            pipeline.fit(x_train, y_train)
            score = rmse(y_val, pipeline.predict(x_val))
            mlflow.log_metrics({"val_rmse": score, "baseline_rmse": baseline})
            mlflow.log_dict(
                {"trial": trial, "train": train_stats, "validation": val_stats,
                 "feature_columns": FEATURE_COLUMNS, "validation_rmse": score,
                 "baseline_rmse": baseline, "observed_drift": drift_report},
                "reports/trial_summary.json",
            )
            if score < best_rmse:
                best_rmse, best_params = score, params
                best_model, best_run = pipeline, run.info.run_id
            print(f"Trial {trial:02d}/{len(candidates)} | val_RMSE=${score:.4f}")

    existing = champion_version(MODEL_NAME, MODEL_ALIAS)
    champion_rmse = None
    if existing is not None:
        previous_model = load_champion(MODEL_NAME, MODEL_ALIAS)
        champion_rmse = rmse(y_val, previous_model.predict(x_val))
    elif args.require_champion:
        raise RuntimeError("No champion alias found: bootstrap using the first authorized train")

    gate_passed = champion_rmse is None or regression_passes(best_rmse, champion_rmse)
    report = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "model_name": MODEL_NAME,
        "alias": MODEL_ALIAS,
        "train_month": args.train_month,
        "val_month": args.val_month,
        "train_data": train_stats,
        "observed_drift": drift_report,
        "validation_data": val_stats,
        "trials": len(candidates),
        "best_run_id": best_run,
        "best_params": best_params,
        "best_validation_rmse": best_rmse,
        "baseline_validation_rmse": baseline,
        "champion_validation_rmse": champion_rmse,
        "regression_gate_passed": gate_passed,
        "promoted": False,
        "registered_version": None,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Report: {args.report}")

    if not gate_passed:
        raise RuntimeError(
            f"REGRESSION GATE FAILED: candidate RMSE {best_rmse:.4f} "
            f"> champion RMSE {champion_rmse:.4f} on the SAME validation sample"
        )

    if not args.no_promote:
        with mlflow.start_run(run_name="selected-model"):
            mlflow.log_params({**best_params, "train_month": args.train_month,
                               "val_month": args.val_month, "seed": args.seed})
            mlflow.log_metrics({"val_rmse": best_rmse, "baseline_rmse": baseline})
            mlflow.log_dict(report, "reports/model_selection.json")
            example = x_val.head(5)
            signature = infer_signature(example, best_model.predict(example))
            model_info = mlflow.sklearn.log_model(
                sk_model=best_model,
                name="model",
                input_example=example,
                signature=signature,
            )
        registered = mlflow.register_model(model_info.model_uri, MODEL_NAME)
        client = MlflowClient()
        client.set_model_version_tag(
            MODEL_NAME, registered.version, "validated_rmse", str(best_rmse)
        )
        client.set_model_version_tag(MODEL_NAME, registered.version, "val_month", args.val_month)
        client.set_registered_model_alias(MODEL_NAME, MODEL_ALIAS, registered.version)
        report["registered_version"] = str(registered.version)
        report["promoted"] = True
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Promoted models:/{MODEL_NAME}@{MODEL_ALIAS} => v{registered.version}")
    else:
        print("Gate passed. Candidate not promoted (--no-promote).")
    return report


if __name__ == "__main__":
    main()
