#!/usr/bin/env bash
# Run the Ibis tests. Two environment variables are considered:
# - PYTEST_BACKENDS: Space-separated list of backends to run

set -eo pipefail

TESTS_DIRS=()

if [ -n "$PYTEST_BACKENDS" ]; then
    TESTS_DIRS+=("ibis/backends/tests")
fi

for backend in $PYTEST_BACKENDS; do
    backend_test_dir="ibis/backends/$backend/tests"
    if [ -d "$backend_test_dir" ]; then
        TESTS_DIRS+=("$backend_test_dir")
    fi
done

set -x

pytest "${TESTS_DIRS[@]}" \
    -q \
    -ra \
    --junitxml=junit.xml \
    --cov=ibis \
    --cov-report=xml:coverage.xml "$@"
