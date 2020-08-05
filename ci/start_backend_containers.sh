#!/bin/bash -e

BASE_DIR="$(readlink -m $(dirname $0)/..)"
CONTAINERS_TO_START=$@

if [ -n "$CONTAINERS_TO_START" ]; then
    python $BASE_DIR/datamgr.py download
fi

for BACKEND in "$CONTAINERS_TO_START"
do
    docker-compose -f $BASE_DIR/ci/docker-compose.yml up --remove-orphans -d --no-build $BACKEND


    if [[ "$BACKEND" == "impala" ]]; then
        python $BASE_DIR/impalamgr.py load --data
    else
        python $BASE_DIR/datamgr.py $BACKEND
    fi
done

docker-compose -f $BASE_DIR/ci/docker-compose.yml up --remove-orphans -d --no-build waiter ci/dockerize.sh $CONTAINERS_TO_START
DOCKER_CODE=$?
echo "DOCKER_CODE: ${DOCKER_CODE}"

if [ $DOCKER_CODE -eq 0 ]
then
  echo "[II] Done."
else
  for s in ${CONTAINERS_TO_START}
  do
    docker container ls
    echo "=============================================================="
    echo "docker ${s} log"
    echo "=============================================================="
    docker logs --details $(docker ps -aqf "name=ci_${s}_1")
  done

fi
exit $DOCKER_CODE
