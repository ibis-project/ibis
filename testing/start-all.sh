#!/usr/bin/env bash

export IBIS_TEST_IMPALA_HOST=localhost
export IBIS_TEST_IMPALA_PORT=21050
export IBIS_TEST_NN_HOST=localhost
export IBIS_TEST_WEBHDFS_PORT=50070
export IBIS_TEST_WEBHDFS_USER=ubuntu

docker-compose up -d
python datamgr.py download
python datamgr.py sqlite
python datamgr.py postgres
python datamgr.py clickhouse

python impalamgr.py load --data --no-udf --data-dir ibis-testing-data
