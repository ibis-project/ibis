#!/bin/sh

# install ibis
python3 -m pip install ipython
POETRY_DYNAMIC_VERSIONING=false python3 -m pip install -e '.[duckdb,clickhouse,examples,geospatial]'
