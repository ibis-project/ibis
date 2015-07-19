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

# This script calls machinery that initializes an ibis.tests.util.IbisTestEnv,
# so it needs those variables set correctly.  It also assumes that WORKSPACE is
# set (i.e., that it is being run as a Jenkins job).  If the latter is not
# true, you can instead set GIT_URL and GIT_BRANCH to check them out manually.

set -e
set -x

printenv

mkdir -p /tmp/impyla-dbapi
TMP_DIR=$(mktemp -d -p /tmp/impyla-dbapi tmpXXXX)

function cleanup {
    rm -rf $TMP_DIR
}
trap cleanup EXIT

cd $TMP_DIR

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

# Setup Python
curl https://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh > miniconda.sh
bash miniconda.sh -b -p $TMP_DIR/miniconda
export PATH="$TMP_DIR/miniconda/bin:$PATH"
conda update -y -q conda
conda info -a

# Install ibis and deps into new environment
CONDA_ENV_NAME=pyenv-ibis-test
conda create -y -q -n $CONDA_ENV_NAME python=$PYTHON_VERSION numpy pandas
source activate $CONDA_ENV_NAME
pip install $IBIS_HOME

python --version
which python

cd $IBIS_HOME

python -c "from ibis.tests.util import IbisTestEnv; print(IbisTestEnv())"

# load necessary test data
scripts/load_test_data.py

# run the test suite
py.test --e2e ibis

# cleanup
scripts/cleanup_testing_data.py
