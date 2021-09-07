#!/usr/bin/env bash
# Run the Ibis tests. Two environment variables are considered:
# - PYTEST_BACKENDS: Space-separated list of backends to run

set -eo pipefail

TESTS_DIRS="ibis/tests ibis/backends/tests"
for BACKEND in $PYTEST_BACKENDS; do
    if [[ -d ibis/backends/$BACKEND/tests ]]; then
        TESTS_DIRS="$TESTS_DIRS ibis/backends/$BACKEND/tests"
    fi
done

set -u

echo "TESTS_DIRS: $TESTS_DIRS"

pytest --version

pytest $TESTS_DIRS \
    -q \
    -ra \
    --junitxml=junit.xml \
    --cov=ibis \
    --cov-report=xml:coverage.xml "$@" | tee pytest.log
