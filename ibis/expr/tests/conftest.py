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

from ibis.expr.tests.mocks import MockConnection

import pytest
import ibis


@pytest.fixture
def schema():
    return [
        ('a', 'int8'),
        ('b', 'int16'),
        ('c', 'int32'),
        ('d', 'int64'),
        ('e', 'float'),
        ('f', 'double'),
        ('g', 'string'),
        ('h', 'boolean'),
    ]


@pytest.fixture
def schema_dict(schema):
    return dict(schema)


@pytest.fixture
def table(schema):
    return ibis.table(schema, name='schema')


@pytest.fixture(params=list('abcdh'))
def int_col(request):
    return request.param


@pytest.fixture(params=list('h'))
def bool_col(request):
    return request.param


@pytest.fixture(params=list('ef'))
def float_col(request):
    return request.param


@pytest.fixture(params=list('abcdefh'))
def numeric_col(request):
    return request.param


@pytest.fixture(params=list('abcdefgh'))
def col(request):
    return request.param


@pytest.fixture
def con():
    return MockConnection()
