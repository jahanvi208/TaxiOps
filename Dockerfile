# syntax=docker/dockerfile:1
FROM python:3.11-slim AS wheels
WORKDIR /build
COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m pip wheel --no-cache-dir --wheel-dir /wheels .

FROM python:3.11-slim AS runtime
# XGBoost native libraries need OpenMP support on slim Debian images.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 tar && \
    rm -rf /var/lib/apt/lists/*
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1
RUN groupadd --gid 10001 taxiops && \
    useradd --uid 10001 --gid 10001 --create-home taxiops && \
    mkdir -p /mlflow && chown 10001:10001 /mlflow
COPY --from=wheels /wheels /wheels
RUN python -m pip install --no-cache-dir /wheels/*.whl && rm -rf /wheels
WORKDIR /app
COPY --chown=taxiops:taxiops mlflow_auth.ini /app/mlflow_auth.ini
USER taxiops
EXPOSE 8000
CMD ["sh", "-c", "exec uvicorn taxiops.api.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
