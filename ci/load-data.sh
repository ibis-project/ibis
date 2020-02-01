#!/bin/bash -e

CWD="$(dirname "${0}")"

python $CWD/datamgr.py download

for arg in $@; do
    if [[ "${arg}" == "impala" ]]; then
	    python "${CWD}"/impalamgr.py load --data &
    else
	    python "${CWD}"/datamgr.py ${arg} &
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
