from __future__ import annotations

import os

import pandas as pd
import pytest
from pytest import param

import ibis
import ibis.expr.datatypes as dt
import ibis.expr.types as ir
from ibis.tests.util import assert_equal

pytest.importorskip("psycopg2")
sa = pytest.importorskip("sqlalchemy")

from sqlalchemy.dialects import postgresql  # noqa: E402

RISINGWAVE_TEST_DB = os.environ.get("IBIS_TEST_RISINGWAVE_DATABASE", "dev")
IBIS_RISINGWAVE_HOST = os.environ.get("IBIS_TEST_RISINGWAVE_HOST", "localhost")
IBIS_RISINGWAVE_PORT = os.environ.get("IBIS_TEST_RISINGWAVE_PORT", "4566")
IBIS_RISINGWAVE_USER = os.environ.get("IBIS_TEST_RISINGWAVE_USER", "root")
IBIS_RISINGWAVE_PASS = os.environ.get("IBIS_TEST_RISINGWAVE_PASSWORD", "")


def test_table(alltypes):
    assert isinstance(alltypes, ir.Table)


def test_array_execute(alltypes):
    d = alltypes.limit(10).double_col
    s = d.execute()
    assert isinstance(s, pd.Series)
    assert len(s) == 10


def test_literal_execute(con):
    expr = ibis.literal("1234")
    result = con.execute(expr)
    assert result == "1234"


def test_simple_aggregate_execute(alltypes):
    d = alltypes.double_col.sum()
    v = d.execute()
    assert isinstance(v, float)


def test_list_tables(con):
    assert con.list_tables()
    assert len(con.list_tables(like="functional")) == 1


def test_compile_toplevel(snapshot):
    t = ibis.table([("foo", "double")], name="t0")

    expr = t.foo.sum()
    result = ibis.postgres.compile(expr)
    snapshot.assert_match(str(result), "out.sql")


def test_list_databases(con):
    assert RISINGWAVE_TEST_DB is not None
    assert RISINGWAVE_TEST_DB in con.list_databases()


def test_schema_type_conversion(con):
    typespec = [
        # name, type, nullable
        ("jsonb", postgresql.JSONB, True, dt.JSON),
    ]

    sqla_types = []
    ibis_types = []
    for name, t, nullable, ibis_type in typespec:
        sqla_types.append(sa.Column(name, t, nullable=nullable))
        ibis_types.append((name, ibis_type(nullable=nullable)))

    # Create a table with placeholder stubs for JSON, JSONB, and UUID.
    table = sa.Table("tname", sa.MetaData(), *sqla_types)

    # Check that we can correctly create a schema with dt.any for the
    # missing types.
    schema = con._schema_from_sqla_table(table)
    expected = ibis.schema(ibis_types)

    assert_equal(schema, expected)


@pytest.mark.parametrize("params", [{}, {"database": RISINGWAVE_TEST_DB}])
def test_create_and_drop_table(con, temp_table, params):
    sch = ibis.schema(
        [
            ("first_name", "string"),
            ("last_name", "string"),
            ("department_name", "string"),
            ("salary", "float64"),
        ]
    )

    con.create_table(temp_table, schema=sch, **params)
    assert con.table(temp_table, **params) is not None

    con.drop_table(temp_table, **params)

    with pytest.raises(sa.exc.NoSuchTableError):
        con.table(temp_table, **params)


@pytest.mark.parametrize(
    ("pg_type", "expected_type"),
    [
        param(pg_type, ibis_type, id=pg_type.lower())
        for (pg_type, ibis_type) in [
            ("boolean", dt.boolean),
            ("bytea", dt.binary),
            ("bigint", dt.int64),
            ("smallint", dt.int16),
            ("integer", dt.int32),
            ("text", dt.string),
            ("real", dt.float32),
            ("double precision", dt.float64),
            ("character varying", dt.string),
            ("date", dt.date),
            ("time", dt.time),
            ("time without time zone", dt.time),
            ("timestamp without time zone", dt.timestamp),
            ("timestamp with time zone", dt.Timestamp("UTC")),
            ("interval", dt.Interval("s")),
            ("numeric", dt.decimal),
            ("jsonb", dt.json),
        ]
    ],
)
def test_get_schema_from_query(con, pg_type, expected_type):
    name = con._quote(ibis.util.guid())
    with con.begin() as c:
        c.exec_driver_sql(f"CREATE TABLE {name} (x {pg_type}, y {pg_type}[])")
    expected_schema = ibis.schema(dict(x=expected_type, y=dt.Array(expected_type)))
    result_schema = con._get_schema_using_query(f"SELECT x, y FROM {name}")
    assert result_schema == expected_schema
    with con.begin() as c:
        c.exec_driver_sql(f"DROP TABLE {name}")


@pytest.mark.xfail(reason="unsupported insert with CTEs")
def test_insert_with_cte(con):
    X = con.create_table("X", schema=ibis.schema(dict(id="int")), temp=False)
    expr = X.join(X.mutate(a=X["id"] + 1), ["id"])
    Y = con.create_table("Y", expr, temp=False)
    assert Y.execute().empty
    con.drop_table("Y")
    con.drop_table("X")


def test_connect_url_with_empty_host():
    con = ibis.connect("risingwave:///dev")
    assert con.con.url.host is None
