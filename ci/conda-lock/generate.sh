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

linux_osx_extras=()
if [ "${python_version}" != "3.11" ]; then
  # clickhouse cityhash doesn't exist for python 3.11
  linux_osx_extras+=(-e clickhouse)
fi

conda lock \
  --file pyproject.toml \
  --file "${python_version_file}" \
  --kind explicit \
  --platform linux-64 \
  --platform osx-64 \
  --filename-template "${template}" \
  --filter-extras \
  --mamba \
  --category dev --category test --category docs \
  "${extras[@]}" "${linux_osx_extras[@]}" -e datafusion

conda lock \
  --file pyproject.toml \
  --file "${python_version_file}" \
  --kind explicit \
  --platform osx-arm64 \
  --platform win-64 \
  --filename-template "${template}" \
  --filter-extras \
  --mamba \
  --category dev --category test --category docs \
  "${extras[@]}"
