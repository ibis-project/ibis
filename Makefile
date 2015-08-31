SHELL := /bin/bash

all:
	python setup.py build_ext --inplace

impala-test:
	pushd scripts && python load_test_data.py --udf && popd

clean-pyc:
	find . -name "*.pyc" -exec rm -rf {} \;
