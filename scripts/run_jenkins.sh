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

printenv

IBIS_HOME=$WORKSPACE
cd $IBIS_HOME

# set up python environment
VENV_NAME=pyvenv-$BUILD_TAG
virtualenv $VENV_NAME && source $VENV_NAME/bin/activate
pip install -U pip setuptools
pip install .

# load necessary test data
$IBIS_HOME/scripts/load_test_data.py

# run the test suite
py.test --e2e ibis

# cleanup
$IBIS_HOME/scripts/cleanup_testing_data.py
deactivate && rm -rf $VENV_NAME
