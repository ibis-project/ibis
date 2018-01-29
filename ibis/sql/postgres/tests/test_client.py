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
import pandas as pd

from ibis.tests.util import assert_equal
import ibis.expr.types as ir
import ibis

pytest.importorskip('sqlalchemy')
pytest.importorskip('psycopg2')

pytestmark = pytest.mark.postgresql

POSTGRES_TEST_DB = os.environ.get('IBIS_TEST_POSTGRES_DATABASE',
                                  'ibis_testing')
IBIS_POSTGRES_HOST = os.environ.get('IBIS_TEST_POSTGRES_HOST',
                                    'localhost')
IBIS_POSTGRES_USER = os.environ.get('IBIS_TEST_POSTGRES_USER',
                                    'postgres')
IBIS_POSTGRES_PASS = os.environ.get('IBIS_TEST_POSTGRES_PASSWORD',
                                    'postgres')


def test_table(alltypes):
    assert isinstance(alltypes, ir.TableExpr)


def test_array_execute(alltypes):
    d = alltypes.limit(10).double_col
    s = d.execute()
    assert isinstance(s, pd.Series)
    assert len(s) == 10


def test_literal_execute(con):
    expr = ibis.literal('1234')
    result = con.execute(expr)
    assert result == '1234'


def test_simple_aggregate_execute(alltypes):
    d = alltypes.double_col.sum()
    v = d.execute()
    assert isinstance(v, float)


def test_list_tables(con):
    assert len(con.list_tables()) > 0
    assert len(con.list_tables(like='functional')) == 1


def test_compile_verify(alltypes):
    unsupported_expr = alltypes.double_col.approx_median()
    assert not unsupported_expr.verify()

    supported_expr = alltypes.double_col.sum()
    assert supported_expr.verify()


def test_database_layer(con, alltypes):
    db = con.database()
    t = db.functional_alltypes

    assert_equal(t, alltypes)

    assert db.list_tables() == con.list_tables()

    db_schema = con.schema("information_schema")

    assert db_schema.list_tables() != con.list_tables()


def test_compile_toplevel():
    t = ibis.table([('foo', 'double')], name='t0')

    # it works!
    expr = t.foo.sum()
    result = ibis.postgres.compile(expr)
    expected = "SELECT sum(t0.foo) AS sum \nFROM t0 AS t0"  # noqa

    assert str(result) == expected


def test_list_databases(con):
    assert POSTGRES_TEST_DB is not None
    assert POSTGRES_TEST_DB in con.list_databases()


def test_list_schemas(con):
    assert 'public' in con.list_schemas()
    assert 'information_schema' in con.list_schemas()


def test_metadata_is_per_table():
    con = ibis.postgres.connect(
        host=IBIS_POSTGRES_HOST,
        database=POSTGRES_TEST_DB,
        user=IBIS_POSTGRES_USER,
        password=IBIS_POSTGRES_PASS,
    )
    assert len(con.meta.tables) == 0

    # assert that we reflect only when a table is requested
    t = con.table('functional_alltypes')  # noqa
    assert 'functional_alltypes' in con.meta.tables
    assert len(con.meta.tables) == 1


def test_schema_table():
    con = ibis.postgres.connect(
        host=IBIS_POSTGRES_HOST,
        database=POSTGRES_TEST_DB,
        user=IBIS_POSTGRES_USER,
        password=IBIS_POSTGRES_PASS,
    )

    # ensure that we can reflect the information schema (which is guaranteed
    # to exist)
    schema = con.schema('information_schema')

    assert isinstance(schema['tables'], ir.TableExpr)
