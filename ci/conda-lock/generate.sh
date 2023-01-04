#!/usr/bin/env bash

set -euo pipefail

python_version="${1}"
python_version_file="$(mktemp --suffix=.yml)"

{
  echo 'name: conda-lock'
  echo 'dependencies:'
  echo "  - python=${python_version}"
} > "${python_version_file}"

extras=(
  -e bigquery
  -e dask
  -e duckdb
  -e impala
  -e mssql
  -e mysql
  -e pandas
  -e polars
  -e postgres
  -e pyspark
  -e snowflake
  -e sqlite
  -e trino
  -e visualization
  -e decompiler
)
template="conda-lock/{platform}-${python_version}.lock"
conda lock \
  --file pyproject.toml \
  --file "${python_version_file}" \
  --kind explicit \
  --platform linux-64 \
  --platform osx-64 \
  --filename-template "${template}" \
  --filter-extras \
  --mamba \
  "${extras[@]}" -e clickhouse

conda lock \
  --file pyproject.toml \
  --file "${python_version_file}" \
  --kind explicit \
  --platform osx-arm64 \
  --platform win-64 \
  --filename-template "${template}" \
  --filter-extras \
  --mamba \
  "${extras[@]}"
