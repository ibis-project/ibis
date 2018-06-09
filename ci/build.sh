#!/bin/bash -e

docker-compose rm --force --stop
docker-compose up -d --no-build mapd postgres mysql clickhouse impala
docker-compose run --rm waiter
docker-compose build --pull ibis
docker-compose run -e LOGLEVEL --rm ibis ci/load-data.sh
