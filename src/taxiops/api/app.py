"""Registry-backed REST API with validated requests and Prometheus telemetry."""
from __future__ import annotations

import logging
import time
from functools import lru_cache

import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, ConfigDict, Field
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from taxiops.config import FEATURE_COLUMNS, MODEL_ALIAS, MODEL_NAME
from taxiops.registry import load_champion

logger = logging.getLogger(__name__)
app = FastAPI(title="TaxiOps", version="0.1.0", description="NYC taxi base-fare estimate API")
API_REQUESTS = Counter(
    "taxiops_http_requests_total", "API HTTP responses", ["method", "path", "status"]
)
API_LATENCY = Histogram(
    "taxiops_http_request_duration_seconds", "API request duration in seconds",
    ["method", "path"],
)


class TripRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    trip_distance: float = Field(ge=0.1, le=100, description="Estimated trip distance in miles")
    passenger_count: int = Field(ge=1, le=8)
    pickup_zone_id: int = Field(ge=1, le=265)
    dropoff_zone_id: int = Field(ge=1, le=265)
    pickup_hour: int = Field(ge=0, le=23)
    pickup_weekday: int = Field(ge=0, le=6, description="Monday=0, Sunday=6")
    pickup_month: int = Field(ge=1, le=12)


class PredictionResponse(BaseModel):
    estimated_fare_usd: float
    model_name: str
    model_alias: str
    unit: str = "USD"


@lru_cache(maxsize=1)
def get_model():
    return load_champion(MODEL_NAME, MODEL_ALIAS)


@app.middleware("http")
async def track_requests(request: Request, call_next):
    start = time.perf_counter()
    # Only registered route templates, not arbitrary user-provided URLs.
    known_routes = {"/predict", "/health", "/metrics"}
    route_path = request.url.path if request.url.path in known_routes else "other"
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        API_REQUESTS.labels(request.method, route_path, str(status)).inc()
        API_LATENCY.labels(request.method, route_path).observe(time.perf_counter() - start)


@app.get("/health")
def health():
    loaded = bool(get_model.cache_info().currsize)
    return {"status": "ok" if loaded else "awaiting_model", "model_loaded": loaded}


@app.get("/metrics", include_in_schema=False)
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictionResponse)
def predict(trip: TripRequest):
    try:
        model = get_model()
    except Exception as exc:
        logger.exception("MLflow registry model could not be loaded")
        raise HTTPException(
            status_code=503, detail="Champion model unavailable. Check MLflow and training."
        ) from exc
    frame = pd.DataFrame([trip.model_dump()], columns=FEATURE_COLUMNS)
    try:
        raw_prediction = float(model.predict(frame)[0])
        if not 0 <= raw_prediction < 1_000_000:
            raise ValueError("Model output is out of range")
    except Exception as exc:
        logger.exception("Inference failed")
        raise HTTPException(status_code=503, detail="Model inference failed") from exc
    return PredictionResponse(
        estimated_fare_usd=round(raw_prediction, 2),
        model_name=MODEL_NAME,
        model_alias=MODEL_ALIAS,
    )
