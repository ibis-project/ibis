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

# import getpass
import os
import ibis
import pytest

CLICKHOUSE_HOST = os.environ.get('IBIS_CLICKHOUSE_HOST', 'localhost')
CLICKHOUSE_PORT = int(os.environ.get('IBIS_CLICKHOUSE_PORT', 9000))
CLICKHOUSE_USER = os.environ.get('IBIS_CLICKHOUSE_USER', 'default')
CLICKHOUSE_PASS = os.environ.get('IBIS_CLICKHOUSE_PASS', '')

IBIS_TEST_CLICKHOUSE_DB = os.environ.get('IBIS_TEST_DATA_DB', 'ibis_testing')


@pytest.fixture(scope='module')
def con():
    return ibis.clickhouse.connect(
        host=CLICKHOUSE_HOST,
        user=CLICKHOUSE_USER,
        password=CLICKHOUSE_PASS,
        database=IBIS_TEST_CLICKHOUSE_DB,
    )


@pytest.fixture(scope='module')
def db(con):
    return con.database()


@pytest.fixture(scope='module')
def alltypes(db):
    return db.functional_alltypes


@pytest.fixture(scope='module')
def df(alltypes):
    return alltypes.execute()


# @pytest.fixture(scope='module')
# def at(alltypes):
#     return alltypes.op().sqla_table


@pytest.fixture
def translate():
    from ibis.clickhouse.compiler import ClickhouseExprTranslator
    return lambda expr: ClickhouseExprTranslator(expr).get_result()
