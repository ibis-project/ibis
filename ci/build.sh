#!/bin/bash -e

docker-compose rm --force --stop
docker-compose up -d --no-build postgres mysql clickhouse impala
docker-compose run waiter
docker-compose build --pull ibis
docker-compose run ibis ci/load-data.sh
