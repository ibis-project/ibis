#!/bin/bash -e

BACKENDS=$@

PYTEST_MARKERS=""

for s in ${BACKENDS[@]}
  do
    if [ "$PYTEST_MARKERS" == "" ]
    then
      PYTEST_MARKERS="-m ${PYTEST_MARKERS} ${s}"
    else
      PYTEST_MARKERS="${PYTEST_MARKERS} or ${s}"
    fi
  done

echo "'${PYTEST_MARKERS}'"
