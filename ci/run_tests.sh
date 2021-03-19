#!/bin/bash -e
# Run the Ibis tests. Two environment variables are considered:
# - PYTEST_BACKENDS: Space-separated list of backends to run
# - PYTEST_EXPRESSION: Marker expression, for example "not udf"

TESTS_DIRS="ibis/tests ibis/backends/tests ibis/backends/base*/tests"
for BACKEND in $PYTEST_BACKENDS; do
    if [[ -d ibis/backends/$BACKEND/tests ]]; then
        TESTS_DIRS="$TESTS_DIRS ibis/backends/$BACKEND/tests"
    fi
done

echo "TESTS_DIRS: $TESTS_DIRS"
echo "PYTEST_EXPRESSION: $PYTEST_EXPRESSION"

if [[ "$PYTEST_BACKENDS" == *"spark"* ]]; then
    PYSPARK_VERSION=$(python -c "import pyspark; print(pyspark.__version__)")
    if [[ $PYSPARK_VERSION == '2'* ]]; then
           echo "Set JAVA_HOME to JAVA_HOME_8_X64"
           export JAVA_HOME=$JAVA_HOME_8_X64
    fi
fi

env

pytest $TESTS_DIRS \
    -m "${PYTEST_EXPRESSION}" \
    -ra \
    --junitxml=junit.xml \
    --cov=ibis \
    --cov-report=xml:coverage.xml
