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

import collections

import pytest

import ibis
import ibis.common.exceptions as com
from ibis.tests.expr.mocks import MockConnection


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
        ('i', 'timestamp'),
        ('j', 'date'),
        ('k', 'time'),
    ]


@pytest.fixture
def schema_dict(schema):
    return collections.OrderedDict(schema)


@pytest.fixture
def table(schema):
    return ibis.table(schema, name='table')


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


@pytest.fixture(params=list('g'))
def string_col(request):
    return request.param


@pytest.fixture(params=list('abcdefgh'))
def col(request):
    return request.param


@pytest.fixture
def con():
    return MockConnection()


@pytest.fixture
def alltypes(con):
    return con.table('alltypes')


@pytest.fixture
def functional_alltypes(con):
    return con.table('functional_alltypes')


@pytest.fixture
def lineitem(con):
    return con.table('tpch_lineitem')


@pytest.hookimpl(hookwrapper=True)
def pytest_pyfunc_call(pyfuncitem):
    """Dynamically add an xfail marker for specific backends."""
    outcome = yield
    try:
        outcome.get_result()
    except (
        com.OperationNotDefinedError,
        com.UnsupportedOperationError,
        com.UnsupportedBackendType,
        NotImplementedError,
    ) as e:
        markers = list(pyfuncitem.iter_markers(name="xfail_unsupported"))
        assert (
            len(markers) == 1
        ), "More than one xfail_unsupported marker found on test {}".format(
            pyfuncitem
        )
        pytest.xfail(reason=repr(e))
