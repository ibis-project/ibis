#!/bin/bash

SERVICES=$@

echo "DOCKER_CODE: ${DOCKER_CODE}"
echo "SERVICES: ${SERVICES}"

if [ $DOCKER_CODE -eq 0 ]
then
  echo "[II] Done."
else
  for s in ${SERVICES}
  do
    docker container ls
    echo "=============================================================="
    echo "docker ${s} log"
    echo "=============================================================="
    docker logs --details $(docker ps -aqf "name=ci_${s}_1")
  done

fi
exit $DOCKER_CODE
