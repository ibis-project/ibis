#!/bin/bash -e

SERVICES=$@

PYTEST_MARKERS="-m"

for s in ${SERVICES[@]}
  do
    if [ "$PYTEST_MARKERS" == "-m" ]
    then
      PYTEST_MARKERS="${PYTEST_MARKERS} ${s}"
    else
      PYTEST_MARKERS="${PYTEST_MARKERS} or ${s}"
    fi
  done

echo "'${PYTEST_MARKERS}'"
