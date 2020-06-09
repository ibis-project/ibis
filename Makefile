.PHONY: all clean develop typecheck stop build start load restart init test testmost testfast testparams docclean doc black

SHELL := /bin/bash
MAKEFILE_DIR = $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))

# PYTHON_VERSION and REQUIREMENTS_TAG defines which `./ci/requirements-dev-$PYTHON_VERSION-$REQUIREMENTS_TAG`
# file will be used for creating the ibis image (see for additional info: `./ci/Dockerfile.dev` and
# `./ci/docker-compose.yml`)
# You can use `3.6` or `3.7` for now for the PYTHON_VERSION
PYTHON_VERSION := 3.6
REQUIREMENTS_TAG := "main"

PYTHONHASHSEED := random

# docker specific
COMPOSE_FILE := "$(MAKEFILE_DIR)/ci/docker-compose.yml"
DOCKER := PYTHON_VERSION=$(PYTHON_VERSION) REQUIREMENTS_TAG="$(REQUIREMENTS_TAG)" docker-compose -f $(COMPOSE_FILE)
DOCKER_UP := $(DOCKER) up --remove-orphans -d --no-build
DOCKER_RUN := $(DOCKER) run --rm
DOCKER_BUILD := $(DOCKER) build
DOCKER_STOP := $(DOCKER) rm --force --stop

# command to be executed inside docker container
DOCKER_RUN_COMMAND := echo "you should do 'make docker_run DOCKER_RUN_COMMAND=[you command]'"

# all backends that ibis using
BACKENDS := clickhouse impala kudu-master kudu-tserver mysql omniscidb parquet postgres sqlite

# backends which are implemented as containers and can be launched through the `docker-compose`
SERVICES := omniscidb postgres mysql clickhouse impala kudu-master kudu-tserver

# the variable contains backends for which test datasets can be automatically loaded
LOADS := sqlite parquet postgres clickhouse omniscidb mysql impala

CURRENT_SERVICES := $(shell $(MAKEFILE_DIR)/ci/backends-to-start.sh "$(BACKENDS)" "$(SERVICES)")
CURRENT_LOADS := $(shell $(MAKEFILE_DIR)/ci/backends-to-start.sh "$(BACKENDS)" "$(LOADS)")
WAITER_COMMAND := $(shell $(MAKEFILE_DIR)/ci/dockerize.sh $(CURRENT_SERVICES))

# pytest specific options
PYTEST_MARKERS := $(shell $(MAKEFILE_DIR)/ci/backends-markers.sh $(BACKENDS))
PYTEST_DOCTEST_OPTIONS := --doctest-modules --doctest-ignore-import-errors
PYTEST_OPTIONS :=

REMOVE_COMPILED_PYTHON_SCRIPTS := (find /ibis -name "*.py[co]" -delete > /dev/null 2>&1 || true)

LOGLEVEL := WARNING


## Targets for code checks

typecheck:
	@mypy --ignore-missing-imports $(MAKEFILE_DIR)/ibis

lint:
	flake8

black:
	# check that black formatting would not be applied
	black --check .

check_pre_commit_hooks:
	# check if all pre-commit hooks are passing
	pre-commit run --all-files

## Targets for setup development environment

clean:
	python setup.py clean
	find $(MAKEFILE_DIR) -name '*.pyc' -type f -delete
	rm -rf $(MAKEFILE_DIR)/build $(MAKEFILE_DIR)/dist $(find $(MAKEFILE_DIR) -name __pycache__ -type d)

develop: clean
	python setup.py develop
	pre-commit install


## DOCKER specific targets

# Targets for code checks inside containers

docker_lint: build
	$(DOCKER_RUN) ibis flake8

docker_black: build
	$(DOCKER_RUN) ibis black --check .

docker_check_pre_commit_hooks: build
	# check if all pre-commit hooks are passing inside ibis container
	$(DOCKER_RUN) ibis pre-commit run --all-files

# Targets for manipulating docker's containers

stop:
	# stop all running docker compose services and remove its
	$(DOCKER_STOP) $(CURRENT_SERVICES)

start:
	# start all docker compose services
	$(DOCKER_UP) $(CURRENT_SERVICES)

build:
	# build the ibis image
	$(DOCKER_BUILD) ibis

wait:
	# wait for services to start
	$(DOCKER_RUN) waiter $(WAITER_COMMAND)
	DOCKER_CODE=$(shell echo $$?) ./ci/check-services.sh $(CURRENT_SERVICES)

load:
	# load datasets for testing purpose
	$(DOCKER_RUN) -e LOGLEVEL=$(LOGLEVEL) ibis ./ci/load-data.sh $(CURRENT_LOADS)

restart: stop
	$(MAKE) start
	$(MAKE) wait

init: restart
	$(MAKE) build
	$(MAKE) load

# Targets for running backend specific Ibis tests inside docker's containers
# BACKENDS can be set to choose which tests should be run:
#   make --directory ibis testparallel BACKENDS='omniscidb impala'

test: init
	$(DOCKER_RUN) -e PYTHONHASHSEED="$(PYTHONHASHSEED)" ibis bash -c "${REMOVE_COMPILED_PYTHON_SCRIPTS} && \
		pytest $(PYTEST_DOCTEST_OPTIONS) $(PYTEST_OPTIONS) ${PYTEST_MARKERS} -k 'not test_import_time'"

testnoinit:
	$(DOCKER_RUN) -e PYTHONHASHSEED="$(PYTHONHASHSEED)" ibis bash -c "${REMOVE_COMPILED_PYTHON_SCRIPTS} && \
		pytest $(PYTEST_DOCTEST_OPTIONS) $(PYTEST_OPTIONS) ${PYTEST_MARKERS} -k 'not test_import_time'"

testparallel: init
	$(DOCKER_RUN) -e PYTHONHASHSEED="$(PYTHONHASHSEED)" ibis bash -c "${REMOVE_COMPILED_PYTHON_SCRIPTS} && \
		pytest $(PYTEST_DOCTEST_OPTIONS) $(PYTEST_OPTIONS) ${PYTEST_MARKERS} -n auto -k 'not test_import_time'"

testparallelnoinit:
	$(DOCKER_RUN) -e PYTHONHASHSEED="$(PYTHONHASHSEED)" ibis bash -c "${REMOVE_COMPILED_PYTHON_SCRIPTS} && \
		pytest $(PYTEST_DOCTEST_OPTIONS) $(PYTEST_OPTIONS) ${PYTEST_MARKERS} -n auto -k 'not test_import_time'"

# Shortcut targets for running a subset of the Ibis tests inside docker's containers

testall:
	@echo "You should use make testmain instead"

testmain:
	$(MAKE) testparallelnoinit REQUIREMENTS_TAG="main" PYTEST_MARKERS="-m 'not (udf or spark or pyspark)'"

testmost:
	$(MAKE) testparallelnoinit REQUIREMENTS_TAG="main" PYTEST_MARKERS="-m 'not (udf or impala or hdfs or spark or pyspark)'"

testfast:
	$(MAKE) testparallelnoinit REQUIREMENTS_TAG="main" PYTEST_MARKERS="-m 'not (udf or impala or hdfs or bigquery or spark or pyspark)'"

testpandas:
	$(MAKE) testparallelnoinit REQUIREMENTS_TAG="main" PYTEST_MARKERS="-m pandas"

testpyspark:
	$(MAKE) testparallelnoinit REQUIREMENTS_TAG="pyspark-spark" PYTEST_MARKERS="-m pyspark"

fastopt:
	@echo -m 'not (backend or bigquery or clickhouse or hdfs or impala or kudu or omniscidb or mysql or postgis or postgresql or superuser or udf)'

# Targets for documentation builds

docclean:
	$(DOCKER_RUN) ibis-docs rm -rf /tmp/docs.ibis-project.org

builddoc: build
	# build the ibis-docs image
	$(DOCKER_BUILD) ibis-docs

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

# Targets for run commands inside ibis and ibis-docs containers

docker_run:
	$(DOCKER_RUN) ibis $(DOCKER_RUN_COMMAND)

docker_docs_run:
	$(DOCKER_RUN) ibis-docs $(DOCKER_RUN_COMMAND)
