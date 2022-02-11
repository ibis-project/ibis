#!/usr/bin/env bash

set -euo pipefail

poetry run pytest \
  --durations=25 \
  -ra \
  --junitxml=junit.xml \
  --cov=ibis \
  --cov-report=xml:coverage.xml "$@"
