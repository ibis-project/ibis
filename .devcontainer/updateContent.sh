#!/usr/bin/env bash

apt update
apt install libgdal-dev

# install ibis
python3 -m pip install ipython
python3 -m pip install -e '.[duckdb,clickhouse,examples,geospatial]'
