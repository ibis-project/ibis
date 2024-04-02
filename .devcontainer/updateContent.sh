#!/usr/bin/env bash

# install ibis
python3 -m pip install ipython
python3 -m pip install -e '.[clickhouse,duckdb,clickhouse,examples,geospatial]'
