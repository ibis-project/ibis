.PHONY: all clean-pyc develop lint test docclean docs docserve

SHELL := /bin/bash
ENVKIND := docs
MAKEFILE_DIR = $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
COMPOSE_FILE := "$(MAKEFILE_DIR)/ci/docker-compose.yml"
DOCKER := ENVKIND=$(ENVKIND) docker-compose -f $(COMPOSE_FILE)
DOCKER_RUN := $(DOCKER) run --rm

clean:
	@find . -name '*.pyc' -type f -delete
	@rm -rf $(find . -name __pycache__ -type d)

develop: clean
	@python setup.py develop

lint:
	@flake8

stop:
# stop all running docker compose services
	@$(DOCKER) rm --force --stop

build:
# build the ibis image
	@$(DOCKER) build --pull ibis

start:
# start all docker compose services
	@$(DOCKER) up -d --no-build mapd postgres mysql clickhouse impala kudu-master kudu-tserver
# wait for services to start
	@$(DOCKER_RUN) waiter

load:
	@$(DOCKER_RUN) -e LOGLEVEL ibis ci/load-data.sh

restart: stop
	@$(MAKE) start

init: restart
	@$(MAKE) build
	@$(MAKE) load

test:
	@ENVKIND=$(ENVKIND) $(MAKEFILE_DIR)/ci/test.sh -n auto -m 'not udf'

testmost:
	@ENVKIND=$(ENVKIND) $(MAKEFILE_DIR)/ci/test.sh -n auto -m 'not udf and not impala and not hdfs'

testfast:
	@ENVKIND=$(ENVKIND) $(MAKEFILE_DIR)/ci/test.sh -n auto -m 'not udf and not impala and not bigquery'

docclean:
	@$(DOCKER_RUN) ibis rm -rf /tmp/docs.ibis-project.org

docs: docclean
	@$(DOCKER_RUN) ibis ping -c 1 quickstart.cloudera
	@$(DOCKER_RUN) ibis git clone --branch gh-pages https://github.com/ibis-project/docs.ibis-project.org /tmp/docs.ibis-project.org
	@$(DOCKER_RUN) ibis find /tmp/docs.ibis-project.org -maxdepth 1 ! -wholename /tmp/docs.ibis-project.org \
	    ! -name '*.git' \
	    ! -name '.' \
	    ! -name 'CNAME' \
	    ! -name '*.nojekyll' \
	    -exec rm -rf {} \;
	@$(DOCKER_RUN) ibis sphinx-build -b html docs/source /tmp/docs.ibis-project.org -W -j auto -T
