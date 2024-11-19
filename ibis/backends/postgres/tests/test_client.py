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
import string
from urllib.parse import quote_plus

import hypothesis as h
import hypothesis.strategies as st
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
from ibis.backends.tests.errors import PsycoPg2OperationalError
from ibis.util import gen_name

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
    assert len(con.list_tables(like="functional")) == 1
    assert {"astronauts", "batting", "diamonds"} <= set(con.list_tables())

    _ = con.create_table("tempy", schema=ibis.schema(dict(id="int")), temp=True)

    assert "tempy" in con.list_tables()
    # temp tables only show up when database='public' (or default)
    assert "tempy" not in con.list_tables(database="tiger")


def test_compile_toplevel(assert_sql):
    t = ibis.table([("foo", "double")], name="t0")

    # it works!
    expr = t.foo.sum()
    assert_sql(expr)


def test_list_catalogs(con):
    assert POSTGRES_TEST_DB is not None
    assert POSTGRES_TEST_DB in con.list_catalogs()


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


@pytest.mark.parametrize("params", [{}, {"database": "public"}])
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

    with pytest.raises(com.TableNotFound, match=temp_table):
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
            ("jsonb", dt.jsonb),
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
        "postgresql://postgres:postgres@localhost:5432/ibis_testing?sslmode=allow"
    )
    assert con.current_catalog == "ibis_testing"


def test_port():
    # check that we parse and use the port (and then of course fail cuz it's bogus)
    with pytest.raises(PsycoPg2OperationalError):
        ibis.connect("postgresql://postgres:postgres@localhost:1337/ibis_testing")


@h.given(st.integers(min_value=4, max_value=1000))
def test_pgvector_type_load(con, vector_size):
    """
    CREATE TABLE items (id bigserial PRIMARY KEY, embedding vector(3));
    INSERT INTO items (embedding) VALUES ('[1,2,3]'), ('[4,5,6]');
    """
    t = con.table("items")

    assert t.schema() == ibis.schema(
        {
            "id": dt.int64(nullable=False),
            "embedding": dt.unknown,
        }
    )

    result = ["[1,2,3]", "[4,5,6]"]
    assert t.to_pyarrow().column("embedding").to_pylist() == result

    query = f"""
    DROP TABLE IF EXISTS itemsvrandom;
    CREATE TABLE itemsvrandom (id bigserial PRIMARY KEY, embedding vector({vector_size}));
    """

    with con.raw_sql(query):
        pass

    t = con.table("itemsvrandom")

    assert t.schema() == ibis.schema(
        {
            "id": dt.int64(nullable=False),
            "embedding": dt.unknown,
        }
    )

    con.drop_table("itemsvrandom")


def test_name_dtype(con):
    expected_schema = ibis.schema(
        {
            "f_table_catalog": dt.String(nullable=True),
            "f_table_schema": dt.String(nullable=True),
            "f_table_name": dt.String(nullable=True),
            "f_geometry_column": dt.String(nullable=True),
            "coord_dimension": dt.Int32(nullable=True),
            "srid": dt.Int32(nullable=True),
            "type": dt.String(nullable=True),
        }
    )

    assert con.tables.geometry_columns.schema() == expected_schema


def test_infoschema_dtypes(con):
    # information_schema.views

    #   |----------------------------+----------------|
    #   | table_catalog              | sql_identifier |
    #   | table_schema               | sql_identifier |
    #   | table_name                 | sql_identifier |
    #   | view_definition            | character_data |
    #   | check_option               | character_data |
    #   | is_updatable               | yes_or_no      |
    #   | is_insertable_into         | yes_or_no      |
    #   | is_trigger_updatable       | yes_or_no      |
    #   | is_trigger_deletable       | yes_or_no      |
    #   | is_trigger_insertable_into | yes_or_no      |
    #   |----------------------------+----------------|
    #
    views_schema = ibis.schema(
        {
            "table_catalog": dt.String(nullable=True),
            "table_schema": dt.String(nullable=True),
            "table_name": dt.String(nullable=True),
            "view_definition": dt.String(nullable=True),
            "check_option": dt.String(nullable=True),
            "is_updatable": dt.String(nullable=True),
            "is_insertable_into": dt.String(nullable=True),
            "is_trigger_updatable": dt.String(nullable=True),
            "is_trigger_deletable": dt.String(nullable=True),
            "is_trigger_insertable_into": dt.String(nullable=True),
        }
    )

    assert con.table("views", database="information_schema").schema() == views_schema

    # information_schema.sql_sizing

    #   |-----------------+-----------------|
    #   | sizing_id       | cardinal_number |
    #   | sizing_name     | character_data  |
    #   | supported_value | cardinal_number |
    #   | comments        | character_data  |
    #   |-----------------+-----------------|

    sql_sizing_schema = ibis.schema(
        {
            "sizing_id": dt.UInt64(nullable=True),
            "sizing_name": dt.String(nullable=True),
            "supported_value": dt.UInt64(nullable=True),
            "comments": dt.String(nullable=True),
        }
    )

    assert (
        con.table("sql_sizing", database="information_schema").schema()
        == sql_sizing_schema
    )

    # information_schema.triggers has a `created` field with the custom timestamp type
    triggers_created_schema = ibis.schema({"created": dt.Timestamp()})

    assert (
        con.table("triggers", database="information_schema").select("created").schema()
        == triggers_created_schema
    )


def test_password_with_bracket():
    password = f"{IBIS_POSTGRES_PASS}[]"
    quoted_pass = quote_plus(password)
    url = f"postgres://{IBIS_POSTGRES_USER}:{quoted_pass}@{IBIS_POSTGRES_HOST}:{IBIS_POSTGRES_PORT}/{POSTGRES_TEST_DB}"
    with pytest.raises(
        PsycoPg2OperationalError,
        match=f'password authentication failed for user "{IBIS_POSTGRES_USER}"',
    ):
        ibis.connect(url)


def test_create_geospatial_table_with_srid(con):
    name = gen_name("geospatial")
    column_names = string.ascii_lowercase
    column_types = [
        "Point",
        "LineString",
        "Polygon",
        "MultiLineString",
        "MultiPoint",
        "MultiPolygon",
    ]
    schema_string = ", ".join(
        f"{column} geometry({dtype}, 4326)"
        for column, dtype in zip(column_names, column_types)
    )
    con.raw_sql(f"CREATE TEMP TABLE {name} ({schema_string})")
    schema = con.get_schema(name)
    assert schema == ibis.schema(
        {
            column: getattr(dt, dtype)(srid=4326)
            for column, dtype in zip(column_names, column_types)
        }
    )


@pytest.fixture(scope="module")
def enum_table(con):
    name = gen_name("enum_table")
    con.raw_sql("CREATE TYPE mood AS ENUM ('sad', 'ok', 'happy')")
    con.raw_sql(f"CREATE TEMP TABLE {name} (mood mood)")
    yield name
    con.raw_sql(f"DROP TABLE {name}")
    con.raw_sql("DROP TYPE mood")


def test_enum_table(con, enum_table):
    t = con.table(enum_table)
    assert t.mood.type() == dt.unknown


def test_parsing_oid_dtype(con):
    # Load a table that uses the OID type and check that we map it to Int64
    t = con.table("pg_class", database="pg_catalog")
    assert t.oid.type() == ibis.dtype("int64")
