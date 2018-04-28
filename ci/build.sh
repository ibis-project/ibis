#!/bin/bash -e

docker-compose rm --force --stop
docker-compose up -d --no-build postgres mysql clickhouse impala mapd
docker-compose run --rm waiter
docker-compose build --pull ibis
docker-compose run --rm ibis ci/load-data.sh
