FROM python:3.11.13-slim-bookworm

ARG PREFECT_VERSION=3.4.9

RUN pip install --no-cache-dir "prefect[postgres]==${PREFECT_VERSION}"

ENTRYPOINT ["prefect"]

