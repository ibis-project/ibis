#!/bin/bash -e

export PYTHON_VERSION="3.6"

docker-compose build ibis
docker-compose build ibis-docs

# TODO(kszucs): move the following commands in a single script
docker-compose run --rm ibis-docs ping -c 1 impala
docker-compose run --rm ibis-docs rm -rf /tmp/docs.ibis-project.org
docker-compose run --rm ibis-docs git clone \
    --branch gh-pages \
    https://github.com/ibis-project/docs.ibis-project.org /tmp/docs.ibis-project.org

docker-compose run --rm ibis-docs find /tmp/docs.ibis-project.org -maxdepth 1 ! -wholename /tmp/docs.ibis-project.org ! -name '*.git' ! -name '.' ! -name 'CNAME' ! -name '*.nojekyll' -exec rm -rf {} \;
docker-compose run --rm ibis-docs conda list --export
docker-compose run --rm ibis-docs sphinx-build -b html docs/source /tmp/docs.ibis-project.org -W -T
