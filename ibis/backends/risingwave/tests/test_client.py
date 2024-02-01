from __future__ import annotations

import os

import pandas as pd
import pytest
import sqlglot as sg
from pytest import param

import ibis
import ibis.expr.datatypes as dt
import ibis.expr.types as ir
from ibis.util import gen_name

pytest.importorskip("psycopg2")

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


def test_create_and_drop_table(con, temp_table):
    sch = ibis.schema([("first_name", "string")])

    con.create_table(temp_table, schema=sch)
    assert con.table(temp_table) is not None

    con.drop_table(temp_table)

    assert temp_table not in con.list_tables()


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
            ("timestamp without time zone", dt.Timestamp(scale=6)),
            ("timestamp with time zone", dt.Timestamp("UTC", scale=6)),
            ("interval", dt.Interval("s")),
            ("numeric", dt.decimal),
            ("jsonb", dt.json),
        ]
    ],
)
def test_get_schema_from_query(con, pg_type, expected_type):
    name = sg.table(gen_name("risingwave_temp_table"), quoted=True)
    with con.begin() as c:
        c.execute(f"CREATE TABLE {name} (x {pg_type}, y {pg_type}[])")
    expected_schema = ibis.schema(dict(x=expected_type, y=dt.Array(expected_type)))
    result_schema = con._get_schema_using_query(f"SELECT x, y FROM {name}")
    assert result_schema == expected_schema
    with con.begin() as c:
        c.execute(f"DROP TABLE {name}")


def test_insert_with_cte(con):
    X = con.create_table("X", schema=ibis.schema(dict(id="int")), temp=False)
    expr = X.join(X.mutate(a=X["id"] + 1), ["id"])
    Y = con.create_table("Y", expr, temp=False)
    assert Y.execute().empty
    con.drop_table("Y")
    con.drop_table("X")
