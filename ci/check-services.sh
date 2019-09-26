#!/bin/bash

docker-compose up --exit-code-from waiter waiter 2> /dev/null

DOCKER_CODE=$?

if [ $DOCKER_CODE -eq 0 ]
then
  echo "[II] Done."
else
  for s in 'clickhouse' 'impala' 'kudu-master' 'kudu-tserver' 'mysql' 'omniscidb' 'postgres'
  do
    docker container ls
    echo "=============================================================="
    echo "docker ${s} log"
    echo "=============================================================="
    docker logs --details $(docker ps -aqf "name=docker_${s}_1")
  done

fi
exit $DOCKER_CODE
