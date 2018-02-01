#!/usr/bin/env bash

CWD=$(dirname $0)


python $CWD/datamgr.py download
python $CWD/datamgr.py mysql &
python $CWD/datamgr.py sqlite &
python $CWD/datamgr.py parquet &
python $CWD/datamgr.py postgres &
python $CWD/datamgr.py clickhouse &
python $CWD/impalamgr.py load --data --data-dir $IBIS_TEST_DATA_DIRECTORY &


FAIL=0

for job in `jobs -p`
do
    wait $job || let FAIL+=1
done

if [ $FAIL -eq 0 ]; then
    echo "Done loading to SQLite, Postgres, Clickhouse and Impala"
    exit 0
else
    echo "Failed loading the datasets" >&2
    exit 1
fi
