#!/bin/bash -e
# Run the Ibis tests. Two environment variables are considered:
# - PYTEST_BACKENDS: Space-separated list of backends to run
# - PYTEST_EXPRESSION: Marker expression, for example "not udf"

TESTS_DIRS="ibis/tests ibis/file/tests"
for BACKEND in $PYTEST_BACKENDS; do
    if [[ -d ibis/$BACKEND/tests ]]; then
        TESTS_DIRS="$TESTS_DIRS ibis/$BACKEND/tests"
    fi
    if [[ -d ibis/sql/$BACKEND/tests ]]; then
        TESTS_DIRS="$TESTS_DIRS ibis/sql/$BACKEND/tests"
    fi
done

echo "TESTS_DIRS: $TESTS_DIRS"
echo "PYTEST_EXPRESSION: $PYTEST_EXPRESSION"


pytest $TESTS_DIRS \
    -m "${PYTEST_EXPRESSION}" \
    -ra \
    --junitxml=junit.xml \
    --cov=ibis \
    --cov-report=xml:coverage.xml
