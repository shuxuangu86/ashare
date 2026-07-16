FROM python:3.11.13-slim-bookworm

ARG MLFLOW_VERSION=3.1.1

RUN pip install --no-cache-dir \
    "mlflow==${MLFLOW_VERSION}" \
    "psycopg2-binary==2.9.10"

