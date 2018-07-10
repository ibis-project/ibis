#!/bin/bash -e

# stop all running docker compose services
docker-compose rm --force --stop

# build the ibis image
docker-compose build --pull ibis

# start all docker compose services
docker-compose up -d --no-build \
    mapd postgres mysql clickhouse impala kudu-master kudu-tserver

# wait for services to start
docker-compose run --rm waiter

# load data
docker-compose run -e LOGLEVEL --rm ibis ci/load-data.sh
