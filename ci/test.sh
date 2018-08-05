#!/bin/bash -e

compose_file=$(dirname "$0")/docker-compose.yml

cmd='$(find /ibis -name "*.py[co]" -delete > /dev/null 2>&1 || true) && pytest "$@"'
docker-compose -f $compose_file build --pull ibis
docker-compose -f $compose_file run --rm ibis bash -c "$cmd" -- "$@"
