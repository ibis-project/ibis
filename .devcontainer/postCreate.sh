#!/bin/sh

source /opt/conda/etc/profile.d/conda.sh

# install ibis
python3 -m pip install ipyflow ipython

# avoid using dynamic versioning by grabbing the version from pyproject.toml
POETRY_DYNAMIC_VERSIONING_BYPASS="$(yq '.tool.poetry.version' pyproject.toml)" \
  python3 -m pip install -e '.[clickhouse,datafusion,duckdb,examples]'
