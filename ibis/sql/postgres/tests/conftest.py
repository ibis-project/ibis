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


import os
import pytest

import ibis

PG_USER = os.environ.get(
    'IBIS_TEST_POSTGRES_USER',
    os.environ.get('PGUSER', 'postgres')
)
PG_PASS = os.environ.get(
    'IBIS_TEST_POSTGRES_PASSWORD',
    os.environ.get('PGPASSWORD', 'postgres')
)
PG_HOST = os.environ.get(
    'IBIS_TEST_POSTGRES_HOST',
    os.environ.get('PGHOST', 'localhost')
)
PG_PORT = os.environ.get(
    'IBIS_TEST_POSTGRES_PORT',
    os.environ.get('PGPORT', 5432)
)
IBIS_TEST_POSTGRES_DB = os.environ.get(
    'IBIS_TEST_POSTGRES_DATABASE',
    os.environ.get('PGDATABASE', 'ibis_testing')
)


@pytest.fixture(scope='module')
def con():
    return ibis.postgres.connect(
        host=PG_HOST,
        user=PG_USER,
        password=PG_PASS,
        database=IBIS_TEST_POSTGRES_DB,
        port=PG_PORT,
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


@pytest.fixture(scope='module')
def at(alltypes):
    return alltypes.op().sqla_table


@pytest.fixture
def translate():
    from ibis.sql.postgres.compiler import PostgreSQLDialect
    dialect = PostgreSQLDialect()
    context = dialect.make_context()
    return lambda expr: dialect.translator(expr, context).get_result()
