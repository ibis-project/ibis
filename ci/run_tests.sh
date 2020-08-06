#!/bin/bash -e
# Run the Ibis tests. A pytest marker expression needs to be
# provided as a parameter, to set which tests should run.

PYTEST_EXPRESSION=$1
if [ -z "$PYTEST_EXPRESSION" ]; then
    echo "All tests can't be run at this point."
    echo "Use a pytest expression, for example: ./ci/run_tests \"mysql and postgres\""
    exit 1
fi

pytest ibis \
    -m "${PYTEST_EXPRESSION}" \
    -ra \
    --numprocesses auto \
    --doctest-modules \
    --doctest-ignore-import-errors \
    -k"-compile -connect" \
    --junitxml=junit.xml \
    --cov=ibis \
    --cov-report=xml:coverage.xml
