#!/usr/bin/env bash

export IBIS_TEST_DATA_DIRECTORY=/tmp/ibis-testing-data
export IBIS_TEST_DOWNLOAD_DIRECTORY=/tmp

CWD=$(dirname $0)

python $CWD/datamgr.py download
python $CWD/datamgr.py sqlite &
python $CWD/datamgr.py postgres &
python $CWD/datamgr.py clickhouse &
python $CWD/impalamgr.py load --data --no-udf --data-dir $IBIS_TEST_DATA_DIRECTORY &

wait

echo "Done loading to SQLite, Postgres, Clickhouse and Impala"

