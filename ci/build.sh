#!/bin/bash -e

compose_file=$(dirname "$0")/docker-compose.yml

# stop all running docker compose services
docker-compose -f "$compose_file" rm --force --stop

# build the ibis image
docker-compose -f "$compose_file" build --pull ibis

# start all docker compose services
docker-compose -f "$compose_file" up -d --no-build \
    mapd postgres mysql clickhouse impala kudu-master kudu-tserver

# wait for services to start
docker-compose -f "$compose_file" run --rm waiter

# load data
docker-compose -f "$compose_file" run -e LOGLEVEL --rm ibis ci/load-data.sh
