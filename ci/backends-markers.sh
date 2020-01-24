#!/bin/bash -e

# The script generates a marker for each backend, which `pytest`
# can use to run specific tests.
#
# Usage:
#  $ ./ci/backends-markers.sh param1
# * param1: array of backends
#
# Example:
# markers=`./ci/backends-markers.sh omniscidb impala` && echo $markers
#
# Output:
# '-m omniscidb or impala'

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
