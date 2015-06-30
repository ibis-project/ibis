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

# Build requested Python version
wget https://www.python.org/ftp/python/$PYTHON_VERSION/Python-$PYTHON_VERSION.tgz
tar -xzf Python-$PYTHON_VERSION.tgz && cd Python-$PYTHON_VERSION
./configure --prefix=$TMP_DIR
make && make altinstall

PY_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PY_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
PY_BIN_DIR=$TMP_DIR/bin
PY_EXEC=$PY_BIN_DIR/python$PY_MAJOR.$PY_MINOR

$PY_EXEC --version
which $PY_EXEC

# Install pip and virtualenv
curl https://bootstrap.pypa.io/get-pip.py | $PY_EXEC
$PY_BIN_DIR/pip install virtualenv

cd $TMP_DIR

# Checkout ibis if necessary
if [ -z "$WORKSPACE" ]; then
    : ${GIT_URL:?"GIT_URL is unset"}
    : ${GIT_BRANCH:?"GIT_BRANCH is unset"}
    git clone $GIT_URL && cd ibis
    git checkout origin/$GIT_BRANCH
    IBIS_HOME=$TMP_DIR/impyla
else
    # WORKSPACE is set, so I must be on a Jenkins slave
    IBIS_HOME=$WORKSPACE
fi

# set up python virtualenv
# note: this is not strictly necessary because we're using a custom-build python
VENV_NAME=pyvenv-ibis-test
virtualenv $VENV_NAME && source $VENV_NAME/bin/activate
pip install -U pip setuptools
pip install $IBIS_HOME

cd $IBIS_HOME

python -c "from ibis.tests.util import IbisTestEnv; print(IbisTestEnv())"

# load necessary test data
scripts/load_test_data.py

# run the test suite
py.test --e2e ibis

# cleanup
scripts/cleanup_testing_data.py
deactivate
