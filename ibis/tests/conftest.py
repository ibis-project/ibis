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

from pytest import skip


def pytest_addoption(parser):
    parser.addoption('--impala', action='store_true', default=False,
                     help='Enable the impala (end-to-end) tests')
    parser.addoption('--madlib', action='store_true', default=False,
                     help='Enable the madlib (end-to-end) tests')
    parser.addoption('--skip-udf', action='store_true', default=False,
                     help='Skip tests marked udf')
    parser.addoption('--skip-superuser', action='store_true', default=False,
                     help='Skip tests marked superuser')


def pytest_runtest_setup(item):
    # the test item is marked impala
    if getattr(item.obj, 'impala', None):
        # but --impala option not set
        if not item.config.getoption('--impala'):
            skip('--impala NOT enabled')

    # the test item is marked madlib
    if getattr(item.obj, 'madlib', None):
        # but --madlib option not set
        if not item.config.getoption('--madlib'):
            skip('--madlib NOT enabled')

    if getattr(item.obj, 'udf', None):
        if item.config.getoption('--skip-udf'):
            skip('--skip-udf enabled')

    if getattr(item.obj, 'superuser', None):
        if item.config.getoption('--skip-superuser'):
            skip('--skip-superuser enabled')
