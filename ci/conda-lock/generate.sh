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
  -e clickhouse
  -e dask
  -e druid
  -e duckdb
  # this doesn't work on any platform yet (issues with resolving some google deps)
  # -e geospatial
  -e impala
  -e mssql
  -e mysql
  -e oracle
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

function conda_lock() {
  local platforms
  platforms=(--platform "$1" --platform "$2")
  shift 2
  conda lock \
    --file pyproject.toml \
    --file "${python_version_file}" \
    --kind explicit \
    "${platforms[@]}" \
    --filename-template "${template}" \
    --filter-extras \
    --conda="$(which conda)" \
    --category dev --category test --category docs \
    "${@}"
}

conda_lock linux-64 osx-64 "${extras[@]}" -e datafusion
conda_lock osx-arm64 win-64 "${extras[@]}"
