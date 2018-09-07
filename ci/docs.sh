#!/bin/bash -e

export ENVKIND=docs
export PYTHON_VERSION="3.6"

docker-compose build --pull ibis
docker-compose run --rm ibis ping -c 1 quickstart.cloudera
docker-compose run --rm ibis rm -rf /tmp/docs.ibis-project.org
docker-compose run --rm ibis git clone \
    --branch gh-pages \
    https://github.com/ibis-project/docs.ibis-project.org /tmp/docs.ibis-project.org

docker-compose run --rm ibis find /tmp/docs.ibis-project.org -maxdepth 1 ! -wholename /tmp/docs.ibis-project.org ! -name '*.git' ! -name '.' ! -name 'CNAME' ! -name '*.nojekyll' -exec rm -rf {} \;
docker-compose run --rm ibis conda env export
docker-compose run --rm ibis sphinx-build -b html docs/source /tmp/docs.ibis-project.org -W -T
