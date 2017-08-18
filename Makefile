.PHONY: all clean-pyc develop lint test docclean docs docserve

SHELL := /bin/bash

all:
	python setup.py build_ext --inplace

clean-pyc:
	find . -name "*.pyc" -exec rm -rf {} \;

develop: clean-pyc
	python setup.py develop

lint:
	flake8

test:
	pytest --pyargs ibis -m 'not impala and not hdfs'

docclean:
	$(MAKE) -C docs clean

docs:
	$(MAKE) -C docs html

docserve: docs
	pushd docs/build/html && python -m http.server --bind localhost
