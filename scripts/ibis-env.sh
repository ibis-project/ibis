#!/bin/bash

echo "Preparing environment for Cloudera quickstart VM"
export IBIS_TEST_IMPALA_HOST=quickstart.cloudera
export IBIS_TEST_NN_HOST=quickstart.cloudera
export IBIS_TEST_WEBHDFS_PORT=50070

echo "Preparing environment for Postgres"
export IBIS_POSTGRES_USER=ibis
export IBIS_POSTGRES_PASS=ibis
export IBIS_TEST_POSTGRES_DB=ibis_testing
