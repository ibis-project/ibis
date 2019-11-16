.PHONY: all clean develop typecheck stop build start load restart init test testmost testfast testparams docclean doc black

SHELL := /bin/bash
MAKEFILE_DIR = $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
PYTHON_VERSION := 3.6
PYTHONHASHSEED := "random"
COMPOSE_FILE := "$(MAKEFILE_DIR)/ci/docker-compose.yml"
DOCKER := docker-compose -f $(COMPOSE_FILE)
DOCKER_RUN := PYTHON_VERSION=${PYTHON_VERSION} $(DOCKER) run --rm
PYTEST_OPTIONS :=
SERVICES := omniscidb postgres mysql clickhouse impala kudu-master kudu-tserver

clean:
	python setup.py clean
	find $(MAKEFILE_DIR) -name '*.pyc' -type f -delete
	rm -rf $(MAKEFILE_DIR)/build $(MAKEFILE_DIR)/dist $(find $(MAKEFILE_DIR) -name __pycache__ -type d)

develop: clean
	python setup.py develop
	pre-commit install

typecheck:
	@mypy --ignore-missing-imports $(MAKEFILE_DIR)/ibis

lint:
	flake8

stop:
# stop all running docker compose services
	$(DOCKER) rm --force --stop ${SERVICES}

build:
# build the ibis image
	$(DOCKER) build --pull ibis

start:
# start all docker compose services
	$(DOCKER) up --remove-orphans -d --no-build ${SERVICES}
# wait for services to start
	$(DOCKER_RUN) waiter

load:
	$(DOCKER_RUN) -e LOGLEVEL ibis ci/load-data.sh

restart: stop
	$(MAKE) start

init: restart
	$(MAKE) build
	$(MAKE) load

testparallel:
	PYTHONHASHSEED=${PYTHONHASHSEED} $(MAKEFILE_DIR)/ci/test.sh -n auto -m 'not udf' -k 'not test_import_time' \
	    --doctest-modules --doctest-ignore-import-errors ${PYTEST_OPTIONS}

test:
	PYTHONHASHSEED=${PYTHONHASHSEED} $(MAKEFILE_DIR)/ci/test.sh ${PYTEST_OPTIONS} -k 'not test_import_time' \
	    --doctest-modules --doctest-ignore-import-errors

testmost:
	PYTHONHASHSEED=${PYTHONHASHSEED} $(MAKEFILE_DIR)/ci/test.sh -n auto -m 'not (udf or impala or hdfs)' -k 'not test_import_time' \
	    --doctest-modules --doctest-ignore-import-errors ${PYTEST_OPTIONS}

testfast:
	PYTHONHASHSEED=${PYTHONHASHSEED} $(MAKEFILE_DIR)/ci/test.sh -n auto -m 'not (udf or impala or hdfs or bigquery)' -k 'not test_import_time' \
	    --doctest-modules --doctest-ignore-import-errors ${PYTEST_OPTIONS}

fastopt:
	@echo -m 'not (backend or bigquery or clickhouse or hdfs or impala or kudu or omniscidb or mysql or postgis or postgresql or superuser or udf)'

docclean:
	$(DOCKER_RUN) ibis-docs rm -rf /tmp/docs.ibis-project.org

builddoc:
# build the ibis-docs image
	$(DOCKER) build ibis ibis-docs

doc: builddoc docclean
	$(DOCKER_RUN) ibis-docs ping -c 1 impala
	$(DOCKER_RUN) ibis-docs rm -rf /tmp/docs.ibis-project.org
	$(DOCKER_RUN) ibis-docs git clone --branch gh-pages https://github.com/ibis-project/docs.ibis-project.org /tmp/docs.ibis-project.org --depth 1
	$(DOCKER_RUN) ibis-docs find /tmp/docs.ibis-project.org \
	    -maxdepth 1 \
	    ! -wholename /tmp/docs.ibis-project.org \
	    ! -name '*.git' \
		! -name '.' \
	    ! -name CNAME \
	    ! -name '*.nojekyll' \
	    -exec rm -rf {} \;
	$(DOCKER_RUN) ibis-docs sphinx-build -b html docs/source /tmp/docs.ibis-project.org -W -T

black:
# check that black formatting would not be applied
	black --check .
