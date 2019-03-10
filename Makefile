.PHONY: all clean develop typecheck stop build start load restart init test testmost testfast testparams docclean doc

SHELL := /bin/bash
ENVKIND := docs
MAKEFILE_DIR = $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
COMPOSE_FILE := "$(MAKEFILE_DIR)/ci/docker-compose.yml"
DOCKER := ENVKIND=$(ENVKIND) docker-compose -f $(COMPOSE_FILE)
DOCKER_RUN := $(DOCKER) run --rm
PYTEST_OPTIONS :=
SERVICES := mapd postgres mysql clickhouse impala kudu-master kudu-tserver

clean:
	@python setup.py clean
	@find $(MAKEFILE_DIR) -name '*.pyc' -type f -delete
	@rm -rf $(MAKEFILE_DIR)/build $(MAKEFILE_DIR)/dist \
	    $(find $(MAKEFILE_DIR) -name __pycache__ -type d)

develop: clean
	@python setup.py develop
	pre-commit install

typecheck:
	@mypy --ignore-missing-imports $(MAKEFILE_DIR)/ibis

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
	@$(DOCKER) up -d --no-build ${SERVICES}
# wait for services to start
	@$(DOCKER_RUN) waiter

load:
	@$(DOCKER_RUN) -e LOGLEVEL ibis ci/load-data.sh

restart: stop
	@$(MAKE) start

init: restart
	@$(MAKE) build
	@$(MAKE) load

testparallel:
	@ENVKIND=$(ENVKIND) $(MAKEFILE_DIR)/ci/test.sh -n auto -m 'not udf' \
	    --doctest-modules --doctest-ignore-import-errors ${PYTEST_OPTIONS}

test:
	@ENVKIND=$(ENVKIND) $(MAKEFILE_DIR)/ci/test.sh ${PYTEST_OPTIONS} \
	    --doctest-modules --doctest-ignore-import-errors

testmost:
	@ENVKIND=$(ENVKIND) $(MAKEFILE_DIR)/ci/test.sh -n auto -m 'not (udf or impala or hdfs)' \
	    --doctest-modules --doctest-ignore-import-errors ${PYTEST_OPTIONS}

testfast:
	@ENVKIND=$(ENVKIND) $(MAKEFILE_DIR)/ci/test.sh -n auto -m 'not (udf or impala or hdfs or bigquery)' \
	    --doctest-modules --doctest-ignore-import-errors ${PYTEST_OPTIONS}

testparams:
	@echo 'not (udf or impala or hdfs or postgresql or mysql or mapd or clickhouse)' \
	    --doctest-modules --doctest-ignore-import-errors ${PYTEST_OPTIONS}

docclean:
	@$(DOCKER_RUN) ibis rm -rf /tmp/docs.ibis-project.org

doc: docclean
	@$(DOCKER_RUN) ibis ping -c 1 quickstart.cloudera
	@$(DOCKER_RUN) ibis git clone --branch gh-pages https://github.com/ibis-project/docs.ibis-project.org /tmp/docs.ibis-project.org
	@$(DOCKER_RUN) ibis find /tmp/docs.ibis-project.org -maxdepth 1 ! -wholename /tmp/docs.ibis-project.org \
	    ! -name '*.git' \
	    ! -name '.' \
	    ! -name 'CNAME' \
	    ! -name '*.nojekyll' \
	    -exec rm -rf {} \;
	@$(DOCKER_RUN) ibis sphinx-build -b html docs/source /tmp/docs.ibis-project.org -W -j auto -T
