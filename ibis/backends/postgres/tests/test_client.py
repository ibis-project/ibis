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

import numpy as np
import pandas as pd
import pytest

import ibis
import ibis.backends.base_sqlalchemy.alchemy as alch  # noqa: E402
import ibis.expr.datatypes as dt
import ibis.expr.types as ir
from ibis.tests.util import assert_equal

sa = pytest.importorskip('sqlalchemy')
pytest.importorskip('psycopg2')

pytestmark = pytest.mark.postgres

POSTGRES_TEST_DB = os.environ.get(
    'IBIS_TEST_POSTGRES_DATABASE', 'ibis_testing'
)
IBIS_POSTGRES_HOST = os.environ.get('IBIS_TEST_POSTGRES_HOST', 'localhost')
IBIS_POSTGRES_USER = os.environ.get('IBIS_TEST_POSTGRES_USER', 'postgres')
IBIS_POSTGRES_PASS = os.environ.get('IBIS_TEST_POSTGRES_PASSWORD', 'postgres')


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


def test_schema_type_conversion():
    typespec = [
        # name, type, nullable
        ('json', sa.dialects.postgresql.JSON, True, dt.JSON),
        ('jsonb', sa.dialects.postgresql.JSONB, True, dt.JSONB),
        ('uuid', sa.dialects.postgresql.UUID, True, dt.UUID),
        ('macaddr', sa.dialects.postgresql.MACADDR, True, dt.MACADDR),
        ('inet', sa.dialects.postgresql.INET, True, dt.INET),
    ]

    sqla_types = []
    ibis_types = []
    for name, t, nullable, ibis_type in typespec:
        sqla_type = sa.Column(name, t, nullable=nullable)
        sqla_types.append(sqla_type)
        ibis_types.append((name, ibis_type(nullable=nullable)))

    # Create a table with placeholder stubs for JSON, JSONB, and UUID.
    engine = sa.create_engine('postgresql://')
    table = sa.Table('tname', sa.MetaData(bind=engine), *sqla_types)

    # Check that we can correctly create a schema with dt.any for the
    # missing types.
    schema = alch.schema_from_table(table)
    expected = ibis.schema(ibis_types)

    assert_equal(schema, expected)


def test_interval_films_schema(con):
    t = con.table("films")
    assert t.len.type() == dt.Interval(unit="m")
    assert t.len.execute().dtype == np.dtype("timedelta64[ns]")


@pytest.mark.parametrize(
    ("column", "expected_dtype"),
    [
        # ("a", dt.Interval("Y")),
        # ("b", dt.Interval("M")),
        ("c", dt.Interval("D")),
        ("d", dt.Interval("h")),
        ("e", dt.Interval("m")),
        ("f", dt.Interval("s")),
        # ("g", dt.Interval("M")),
        ("h", dt.Interval("h")),
        ("i", dt.Interval("m")),
        ("j", dt.Interval("s")),
        ("k", dt.Interval("m")),
        ("l", dt.Interval("s")),
        ("m", dt.Interval("s")),
    ],
)
def test_all_interval_types_schema(intervals, column, expected_dtype):
    assert intervals[column].type() == expected_dtype


@pytest.mark.parametrize(
    ("column", "expected_dtype"),
    [
        # ("a", dt.Interval("Y")),
        # ("b", dt.Interval("M")),
        ("c", dt.Interval("D")),
        ("d", dt.Interval("h")),
        ("e", dt.Interval("m")),
        ("f", dt.Interval("s")),
        # ("g", dt.Interval("M")),
        ("h", dt.Interval("h")),
        ("i", dt.Interval("m")),
        ("j", dt.Interval("s")),
        ("k", dt.Interval("m")),
        ("l", dt.Interval("s")),
        ("m", dt.Interval("s")),
    ],
)
def test_all_interval_types_execute(intervals, column, expected_dtype):
    expr = intervals[column]
    series = expr.execute()
    assert series.dtype == np.dtype("timedelta64[ns]")


@pytest.mark.xfail(
    raises=ValueError, reason="Year and month interval types not yet supported"
)
def test_unsupported_intervals(con):
    t = con.table("not_supported_intervals")
    assert t["a"].type() == dt.Interval("Y")
    assert t["b"].type() == dt.Interval("M")
    assert t["g"].type() == dt.Interval("M")


@pytest.mark.parametrize('params', [{}, {'database': POSTGRES_TEST_DB}])
def test_create_and_drop_table(con, temp_table, params):
    sch = ibis.schema(
        [
            ('first_name', 'string'),
            ('last_name', 'string'),
            ('department_name', 'string'),
            ('salary', 'float64'),
        ]
    )

    con.create_table(temp_table, schema=sch, **params)
    assert con.table(temp_table, **params) is not None

    con.drop_table(temp_table, **params)

    with pytest.raises(sa.exc.NoSuchTableError):
        con.table(temp_table, **params)
