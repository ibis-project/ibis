#!/bin/bash -e

cmd='$(find /ibis -name "*.py[co]" -delete > /dev/null 2>&1 || true) && pytest "$@"'
docker-compose build --pull ibis
docker-compose run --rm ibis bash -c "$cmd" -- "$@"
