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

import ibis

groups = ['hdfs', 'impala', 'madlib', 'postgresql', 'sqlite', 'kudu']


def pytest_configure(config):
    if config.getvalue('iverbose'):
        ibis.options.verbose = True


def pytest_addoption(parser):
    for group in groups:
        parser.addoption('--{0}'.format(group), action='store_true',
                         default=False,
                         help=('Enable the {0} (end-to-end) tests'
                               .format(group)))

    for group in groups:
        parser.addoption('--only-{0}'.format(group), action='store_true',
                         default=False,
                         help=('Enable only the {0} (end-to-end) tests'
                               .format(group)))

    parser.addoption('--skip-udf', action='store_true', default=False,
                     help='Skip tests marked udf')
    parser.addoption('--skip-superuser', action='store_true', default=False,
                     help='Skip tests marked superuser')

    parser.addoption('--iverbose', action='store_true', default=False,
                     help='Set Ibis to verbose')


def pytest_runtest_setup(item):
    only_set = False

    for group in groups:
        only_flag = '--only-{0}'.format(group)
        flag = '--{0}'.format(group)

        if item.config.getoption(only_flag):
            only_set = True
        elif getattr(item.obj, group, None):
            if not item.config.getoption(flag):
                skip('{0} NOT enabled'.format(flag))

    if only_set:
        skip_item = True
        for group in groups:
            only_flag = '--only-{0}'.format(group)
            if (getattr(item.obj, group, False) and
                    item.config.getoption(only_flag)):
                skip_item = False

        if skip_item:
            skip('Only running some groups with only flags')

    if getattr(item.obj, 'udf', None):
        if item.config.getoption('--skip-udf'):
            skip('--skip-udf enabled')

    if getattr(item.obj, 'superuser', None):
        if item.config.getoption('--skip-superuser'):
            skip('--skip-superuser enabled')
