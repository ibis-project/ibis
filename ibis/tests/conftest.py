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
    parser.addoption('--e2e', action='store_true', default=False,
                     help='Enable the e2e (end-to-end) tests')


def pytest_runtest_setup(item):
    if getattr(item.obj, 'e2e', None):  # the test item is marked e2e
        if not item.config.getvalue('e2e'): # but --e2e option not set
            skip('e2e was not enabled')
