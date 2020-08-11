#!/bin/bash -e
# Run the Ibis tests. Two environment variables are considered:
# - PYTEST_BACKENDS: Space-separated list of backends to run
# - PYTEST_EXPRESSION: Marker expression, for example "not udf"

# TODO have a more consistent test structure among backends
# so not so many directories need to be checked
TESTS_DIRS="ibis/tests ibis/expr/tests ibis/sql/tests ibis/sql/vertica/tests ibis/sql/presto/tests ibis/sql/redshift/tests"
for BACKEND in $PYTEST_BACKENDS; do
    if [[ -d ibis/$BACKEND/tests ]]; then
        TESTS_DIRS="$TESTS_DIRS ibis/$BACKEND/tests"
    fi
    if [[ -d ibis/sql/$BACKEND/tests ]]; then
        TESTS_DIRS="$TESTS_DIRS ibis/sql/$BACKEND/tests"
    fi
    if [[ -d ibis/$BACKEND/execution/tests ]]; then
        TESTS_DIRS="$TESTS_DIRS ibis/$BACKEND/execution/tests"
    fi
    if [[ -d ibis/$BACKEND/udf/tests ]]; then
        TESTS_DIRS="$TESTS_DIRS ibis/$BACKEND/udf/tests"
    fi
done

echo "TESTS_DIRS: $TESTS_DIRS"
echo "PYTEST_EXPRESSION: $PYTEST_EXPRESSION"


pytest $TESTS_DIRS \
    -m "${PYTEST_EXPRESSION}" \
    -ra \
    --doctest-modules \
    --doctest-ignore-import-errors \
    -k"-compile -connect" \
    --junitxml=junit.xml \
    --cov=ibis \
    --cov-report=xml:coverage.xml
