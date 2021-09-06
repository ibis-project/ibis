#!/bin/bash -e
# Run the Ibis tests. Two environment variables are considered:
# - PYTEST_BACKENDS: Space-separated list of backends to run

TESTS_DIRS="ibis/tests ibis/backends/tests"
for BACKEND in $PYTEST_BACKENDS; do
    if [[ -d ibis/backends/$BACKEND/tests ]]; then
        TESTS_DIRS="$TESTS_DIRS ibis/backends/$BACKEND/tests"
    fi
done

echo "TESTS_DIRS: $TESTS_DIRS"

set -o pipefail

pytest $TESTS_DIRS \
    -q \
    -ra \
    --junitxml=junit.xml \
    --cov=ibis \
    --cov-report=xml:coverage.xml "$@" | tee pytest.log
