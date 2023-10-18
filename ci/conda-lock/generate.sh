#!/usr/bin/env bash

set -euo pipefail

extras=(
  -e bigquery
  -e clickhouse
  -e dask
  -e datafusion
  -e druid
  -e duckdb
  -e geospatial
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
  -e deltalake
)

# directory of this script
top="$(dirname "$(readlink -f -- "$0")")"

python_version="${1}"
shift 1

conda lock \
  --file pyproject.toml \
  --file "${top}/versions/python-${python_version}.yml" \
  --channel conda-forge \
  --platform linux-64 \
  --platform osx-64 \
  --platform osx-arm64 \
  --platform win-64 \
  --filter-extras \
  --mamba \
  --category dev --category test --category docs \
  "${extras[@]}" \
  "${@}"
