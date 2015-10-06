#! /usr/bin/env bash
# Copyright 2015 Cloudera Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This script calls machinery that initializes an
# ibis.impala.tests.common.IbisTestEnv, so it needs those variables set
# correctly.  It also assumes that WORKSPACE is set (i.e., that it is being run
# as a Jenkins job).  If the latter is not true, you can instead set GIT_URL
# and GIT_BRANCH to check them out manually.  For pulling in a pull request,
# set GITHUB_PR.  For reporting to codecov.io, set CODECOV_TOKEN.  Set
# PYTHON_VERSION to specify which version to run the tests on.

set -e
set -x

printenv

mkdir -p /tmp/ibis-tests
TMP_DIR=$(mktemp -d -p /tmp/ibis-tests tmpXXXX)

function cleanup {
    rm -rf $TMP_DIR
}
trap cleanup EXIT

cd $TMP_DIR

# Add LLVM to PATH
if [ -n "$IBIS_TEST_LLVM_CONFIG" ]; then
    export PATH="$($IBIS_TEST_LLVM_CONFIG --bindir):$PATH"
fi

# Checkout ibis if necessary
if [ -z "$WORKSPACE" ]; then
    : ${GIT_URL:?"GIT_URL is unset"}
    : ${GIT_BRANCH:?"GIT_BRANCH is unset"}
    git clone $GIT_URL
    pushd ibis && git checkout origin/$GIT_BRANCH && popd
    IBIS_HOME=$TMP_DIR/ibis
else
    # WORKSPACE is set, so I must be on a Jenkins slave
    IBIS_HOME=$WORKSPACE
fi

# pull in PR if necessary
if [ -z "$WORKSPACE" -a -n "$GITHUB_PR" ]; then
    pushd $IBIS_HOME
    git clean -d -f
    git fetch origin pull/$GITHUB_PR/head:pr_$GITHUB_PR
    git checkout pr_$GITHUB_PR
    popd
fi

pushd $IBIS_HOME && git status && popd

# Setup Python
curl https://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh > miniconda.sh
bash miniconda.sh -b -p $TMP_DIR/miniconda
export PATH="$TMP_DIR/miniconda/bin:$PATH"
conda update -y -q conda
conda info -a

# Install ibis and deps into new environment
CONDA_ENV_NAME=pyenv-ibis-test
conda create -y -q -n $CONDA_ENV_NAME python=$PYTHON_VERSION pip numpy pandas
source activate $CONDA_ENV_NAME
pip install click
pip install pytest-cov
# preempt the requirements.txt file by installing impyla master
pip install git+https://github.com/cloudera/impyla.git
pip install $IBIS_HOME

python --version
which python

if [ $IBIS_TEST_AUTH_MECH = "GSSAPI" -o $IBIS_TEST_AUTH_MECH = "LDAP" ]; then
    sudo yum install -y cyrus-sasl-devel
    pip install requests-kerberos
    pip install git+https://github.com/laserson/python-sasl.git@cython

    # CLOUDERA INTERNAL JENKINS/KERBEROS CONFIG
    kinit -l 4h -kt /cdep/keytabs/hive.keytab hive
    sudo -u hive PYTHON_EGG_CACHE=/dev/null impala-shell -k -q "GRANT ALL ON SERVER TO ROLE cdep_default_admin WITH GRANT OPTION"
    kdestroy
    kinit -l 4h -kt /cdep/keytabs/systest.keytab systest
fi

cd $IBIS_HOME

python -c "from ibis.impala.tests.common import IbisTestEnv; print(IbisTestEnv())"

# load necessary test data (without overwriting)
scripts/test_data_admin.py load --data --no-udf

if [ -z "$WORKSPACE" ]; then
    # on kerberized cluster, skip UDF work
    PYTEST_KERB_ARGS="--skip-udf --skip-superuser"
fi

if [ -n "$WORKSPACE" ]; then
    # non-kerb cluster: build and load the UDFs
    scripts/test_data_admin.py load --no-data --udf --overwrite
fi

PYTEST_COV_ARGS="--cov ibis --cov-report xml --cov-report term --cov-config .coveragerc"

# run the test suite
py.test -vv -r sxX $PYTEST_KERB_ARGS $PYTEST_COV_ARGS --impala --hdfs --sqlite ibis

# cleanup temporary data (but not testing data)
scripts/test_data_admin.py cleanup --tmp-data --tmp-db

# Enforce flake8 etc
if [ $PYTHON_VERSION != "2.6" ]; then
    pip install prospector
    prospector
fi

# Report code coverage to codecov.io
if [ -n $CODECOV_TOKEN ]; then
    bash <(curl -s https://codecov.io/bash) -t $CODECOV_TOKEN
fi
