#!/bin/sh

# install ibis
python3 -m pip install ipython

# avoid using dynamic versioning by grabbing the version from pyproject.toml
POETRY_DYNAMIC_VERSIONING_BYPASS="$(yq '.tool.poetry.version' pyproject.toml)" \
  python3 -m pip install -e '.[duckdb,clickhouse,examples,geospatial]'
