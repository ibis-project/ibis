SHELL := /bin/bash

all:
	python setup.py build_ext --inplace

impala-test:
	pushd scripts && python load_test_data.py --udf && popd

clean-pyc:
	find . -name "*.pyc" -exec rm -rf {} \;

develop: clean-pyc
	python setup.py develop

lint:
	flake8

test:
	pytest --pyargs ibis -m 'not impala and not hdfs'
