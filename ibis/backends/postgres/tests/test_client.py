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
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pandas.testing as tm
import pytest
import sqlglot as sg
from pytest import param

import ibis
import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.types as ir

pytest.importorskip("psycopg2")

POSTGRES_TEST_DB = os.environ.get("IBIS_TEST_POSTGRES_DATABASE", "ibis_testing")
IBIS_POSTGRES_HOST = os.environ.get("IBIS_TEST_POSTGRES_HOST", "localhost")
IBIS_POSTGRES_PORT = os.environ.get("IBIS_TEST_POSTGRES_PORT", "5432")
IBIS_POSTGRES_USER = os.environ.get("IBIS_TEST_POSTGRES_USER", "postgres")
IBIS_POSTGRES_PASS = os.environ.get("IBIS_TEST_POSTGRES_PASSWORD", "postgres")


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
    assert len(con.list_tables()) > 0
    assert len(con.list_tables(like="functional")) == 1


def test_compile_toplevel(snapshot):
    t = ibis.table([("foo", "double")], name="t0")

    # it works!
    expr = t.foo.sum()
    result = ibis.postgres.compile(expr)
    snapshot.assert_match(str(result), "out.sql")


def test_list_databases(con):
    assert POSTGRES_TEST_DB is not None
    assert POSTGRES_TEST_DB in con.list_databases()


def test_interval_films_schema(con):
    t = con.table("films")
    assert t.len.type() == dt.Interval(unit="m")
    assert issubclass(t.len.execute().dtype.type, np.timedelta64)


@pytest.mark.parametrize(
    ("column", "expected_dtype"),
    [
        # a, b and g are variable length intervals, like YEAR TO MONTH
        param("c", dt.Interval("D"), id="day"),
        param("d", dt.Interval("h"), id="hour"),
        param("e", dt.Interval("m"), id="minute"),
        param("f", dt.Interval("s"), id="second"),
    ],
)
def test_all_interval_types_execute(intervals, column, expected_dtype):
    expr = intervals[column]
    assert expr.type() == expected_dtype

    series = expr.execute()
    assert issubclass(series.dtype.type, np.timedelta64)


def test_unsupported_intervals(con):
    t = con.table("not_supported_intervals")
    assert t["a"].type() == dt.Interval("Y")
    assert t["b"].type() == dt.Interval("Y")
    assert t["g"].type() == dt.Interval("M")


@pytest.mark.parametrize("params", [{}, {"database": POSTGRES_TEST_DB}])
def test_create_and_drop_table(con, temp_table, params):
    sch = ibis.schema(
        [
            ("first_name", "string"),
            ("last_name", "string"),
            ("department_name", "string"),
            ("salary", "float64"),
        ]
    )

    t = con.create_table(temp_table, schema=sch, **params)
    assert t is not None
    assert con.table(temp_table, **params) is not None

    con.drop_table(temp_table, **params)

    with pytest.raises(com.IbisError):
        con.table(temp_table, **params)


@pytest.mark.parametrize(
    ("pg_type", "expected_type"),
    [
        param(pg_type, ibis_type, id=pg_type.lower())
        for (pg_type, ibis_type) in [
            ("boolean", dt.boolean),
            ("bytea", dt.binary),
            ("char", dt.string),
            ("bigint", dt.int64),
            ("smallint", dt.int16),
            ("integer", dt.int32),
            ("text", dt.string),
            ("json", dt.json),
            ("point", dt.point),
            ("polygon", dt.polygon),
            ("line", dt.linestring),
            ("real", dt.float32),
            ("double precision", dt.float64),
            ("macaddr", dt.macaddr),
            ("macaddr8", dt.macaddr),
            ("inet", dt.inet),
            ("character", dt.string),
            ("character varying", dt.string),
            ("date", dt.date),
            ("time", dt.time),
            ("time without time zone", dt.time),
            ("timestamp without time zone", dt.Timestamp(scale=6)),
            ("timestamp with time zone", dt.Timestamp("UTC", scale=6)),
            ("interval", dt.Interval("s")),
            ("numeric", dt.decimal),
            ("numeric(3, 2)", dt.Decimal(3, 2)),
            ("uuid", dt.uuid),
            ("jsonb", dt.json),
            ("geometry", dt.geometry),
            ("geography", dt.geography),
        ]
    ],
)
def test_get_schema_from_query(con, pg_type, expected_type):
    name = sg.table(ibis.util.guid()).sql("postgres")
    with con._safe_raw_sql(f"CREATE TEMP TABLE {name} (x {pg_type}, y {pg_type}[])"):
        pass
    expected_schema = ibis.schema(dict(x=expected_type, y=dt.Array(expected_type)))
    result_schema = con._get_schema_using_query(f"SELECT x, y FROM {name}")
    assert result_schema == expected_schema


@pytest.mark.parametrize("col", ["search", "simvec"])
def test_unknown_column_type(con, col):
    awards_players = con.table("awards_players_special_types")
    assert awards_players[col].type().is_unknown()


def test_insert_with_cte(con):
    X = con.create_table("X", schema=ibis.schema(dict(id="int")), temp=True)
    assert "X" in con.list_tables()
    expr = X.join(X.mutate(a=X["id"] + 1), ["id"])
    Y = con.create_table("Y", expr, temp=True)
    assert Y.execute().empty


@pytest.fixture(scope="module")
def contz(con):
    with con.begin() as c:
        c.execute("SHOW TIMEZONE")
        [(tz,)] = c.fetchall()
        c.execute("SET TIMEZONE TO 'America/New_York'")

    yield con

    with con.begin() as c:
        c.execute(f"SET TIMEZONE TO '{tz}'")


def test_timezone_from_column(contz, snapshot):
    with contz.begin() as c:
        c.execute(
            """
            CREATE TEMPORARY TABLE x (
                id BIGINT,
                ts_tz TIMESTAMP WITH TIME ZONE NOT NULL,
                ts_no_tz TIMESTAMP WITHOUT TIME ZONE NOT NULL
            );

            INSERT INTO x VALUES
                (1, '2018-01-01 00:00:01+00', '2018-01-01 00:00:02');

            CREATE TEMPORARY TABLE y AS SELECT 1::BIGINT AS id;
            """
        )

    case = (
        contz.table("x")
        .rename(tz="ts_tz", no_tz="ts_no_tz")
        .left_join(contz.table("y"), "id")
    )

    result = case.execute()
    expected = pd.DataFrame(
        {
            "id": [1],
            "tz": [pd.Timestamp("2018-01-01 00:00:01", tz="UTC")],
            "no_tz": [pd.Timestamp("2018-01-01 00:00:02")],
            "id_right": [1],
        }
    )
    tm.assert_frame_equal(result, expected)
    snapshot.assert_match(ibis.to_sql(case), "out.sql")


def test_kwargs_passthrough_in_connect():
    con = ibis.connect(
        "postgresql://postgres:postgres@localhost/ibis_testing?sslmode=allow"
    )
    assert con.current_database == "ibis_testing"
