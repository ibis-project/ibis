#!/usr/bin/env bash

CWD="$(dirname "${0}")"

declare -A argcommands=([sqlite]=sqlite
                        [parquet]="parquet -i"
                        [postgres]=postgres
                        [clickhouse]=clickhouse
                        [mapd]=mapd
                        [mysql]=mysql
                        [impala]=impala
                        [spark]=spark)

if [[ "$#" == 0 ]]; then
    ARGS=(${!argcommands[@]})  # keys of argcommands
else
    ARGS=($*)
fi

python $CWD/datamgr.py download

for arg in ${ARGS[@]}; do
    if [[ "${arg}" == "impala" ]]; then
	python "${CWD}"/impalamgr.py load --data &
    else
	python "${CWD}"/datamgr.py ${argcommands[${arg}]} &
    fi
done

FAIL=0

for job in `jobs -p`
do
    wait "${job}" || let FAIL+=1
done

if [[ "${FAIL}" == 0 ]]; then
    exit 0
else
    exit 1
fi
