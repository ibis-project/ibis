from __future__ import annotations

import json
from datetime import date
from operator import methodcaller

import pandas as pd
import pandas.testing as tm
import pytest
import sqlalchemy as sa
from pytest import param
from sqlalchemy.dialects import mysql

import ibis
import ibis.expr.datatypes as dt
from ibis import udf
from ibis.util import gen_name

MYSQL_TYPES = [
    ("tinyint", dt.int8),
    ("int1", dt.int8),
    ("boolean", dt.int8),
    ("smallint", dt.int16),
    ("int2", dt.int16),
    # ("mediumint", dt.int32), => https://github.com/tobymao/sqlglot/issues/2109
    # ("int3", dt.int32), => https://github.com/tobymao/sqlglot/issues/2109
    ("int", dt.int32),
    ("int4", dt.int32),
    ("integer", dt.int32),
    ("bigint", dt.int64),
    ("decimal", dt.Decimal(10, 0)),
    ("decimal(5, 2)", dt.Decimal(5, 2)),
    ("dec", dt.Decimal(10, 0)),
    ("numeric", dt.Decimal(10, 0)),
    ("fixed", dt.Decimal(10, 0)),
    ("float", dt.float32),
    ("double", dt.float64),
    ("timestamp", dt.Timestamp("UTC")),
    ("date", dt.date),
    ("time", dt.time),
    ("datetime", dt.timestamp),
    ("year", dt.int8),
    ("char(32)", dt.string),
    ("char byte", dt.binary),
    ("varchar(42)", dt.string),
    ("mediumtext", dt.string),
    ("text", dt.string),
    ("binary(42)", dt.binary),
    ("varbinary(42)", dt.binary),
    ("bit(1)", dt.int8),
    ("bit(9)", dt.int16),
    ("bit(17)", dt.int32),
    ("bit(33)", dt.int64),
    # mariadb doesn't have a distinct json type
    ("json", dt.string),
    ("enum('small', 'medium', 'large')", dt.string),
    # con.table(name) first parses the type correctly as ibis inet using sqlglot,
    # then convert these types to sqlalchemy types then a sqlalchemy table to
    # get the ibis schema again from the alchemy types, but alchemy doesn't
    # support inet6 so it gets converted to string eventually
    # ("inet6", dt.inet),
    ("set('a', 'b', 'c', 'd')", dt.Array(dt.string)),
    ("mediumblob", dt.binary),
    ("blob", dt.binary),
    ("uuid", dt.uuid),
]


@pytest.mark.parametrize(
    ("mysql_type", "expected_type"),
    [
        param(mysql_type, ibis_type, id=mysql_type)
        for mysql_type, ibis_type in MYSQL_TYPES
    ],
)
def test_get_schema_from_query(con, mysql_type, expected_type):
    raw_name = ibis.util.guid()
    name = con._quote(raw_name)
    expected_schema = ibis.schema(dict(x=expected_type))

    # temporary tables get cleaned up by the db when the session ends, so we
    # don't need to explicitly drop the table
    with con.begin() as c:
        c.exec_driver_sql(f"CREATE TEMPORARY TABLE {name} (x {mysql_type})")

    result_schema = con._get_schema_using_query(f"SELECT * FROM {name}")
    assert result_schema == expected_schema

    t = con.table(raw_name)
    assert t.schema() == expected_schema


@pytest.mark.parametrize("coltype", ["TINYBLOB", "MEDIUMBLOB", "BLOB", "LONGBLOB"])
def test_blob_type(con, coltype):
    tmp = f"tmp_{ibis.util.guid()}"
    with con.begin() as c:
        c.exec_driver_sql(f"CREATE TEMPORARY TABLE {tmp} (a {coltype})")
    t = con.table(tmp)
    assert t.schema() == ibis.schema({"a": dt.binary})


@pytest.fixture(scope="session")
def tmp_t(con_nodb):
    with con_nodb.begin() as c:
        c.exec_driver_sql("CREATE TABLE IF NOT EXISTS test_schema.t (x INET6)")
    yield
    with con_nodb.begin() as c:
        c.exec_driver_sql("DROP TABLE IF EXISTS test_schema.t")


@pytest.mark.usefixtures("setup_privs", "tmp_t")
def test_get_schema_from_query_other_schema(con_nodb):
    t = con_nodb.table("t", schema="test_schema")
    assert t.schema() == ibis.schema({"x": dt.string})


def test_zero_timestamp_data(con):
    sql = """
    CREATE TEMPORARY TABLE ztmp_date_issue
    (
        name      CHAR(10) NULL,
        tradedate DATETIME NOT NULL,
        date      DATETIME NULL
    );
    """
    with con.begin() as c:
        c.exec_driver_sql(sql)
        c.exec_driver_sql(
            """
            INSERT INTO ztmp_date_issue VALUES
                ('C', '2018-10-22', 0),
                ('B', '2017-06-07', 0),
                ('C', '2022-12-21', 0)
            """
        )
    t = con.table("ztmp_date_issue")
    result = t.execute()
    expected = pd.DataFrame(
        {
            "name": ["C", "B", "C"],
            "tradedate": pd.to_datetime(
                [date(2018, 10, 22), date(2017, 6, 7), date(2022, 12, 21)]
            ),
            "date": [pd.NaT, pd.NaT, pd.NaT],
        }
    )
    tm.assert_frame_equal(result, expected)


@pytest.fixture(scope="module")
def enum_t(con):
    name = gen_name("mysql_enum_test")
    t = sa.Table(
        name, sa.MetaData(), sa.Column("sml", mysql.ENUM("small", "medium", "large"))
    )
    with con.begin() as bind:
        t.create(bind=bind)
        bind.execute(t.insert().values(sml="small"))

    yield con.table(name)
    con.drop_table(name, force=True)


@pytest.mark.parametrize(
    ("expr_fn", "expected"),
    [
        (methodcaller("startswith", "s"), pd.Series([True], name="sml")),
        (methodcaller("endswith", "m"), pd.Series([False], name="sml")),
        (methodcaller("re_search", "mall"), pd.Series([True], name="sml")),
        (methodcaller("lstrip"), pd.Series(["small"], name="sml")),
        (methodcaller("rstrip"), pd.Series(["small"], name="sml")),
        (methodcaller("strip"), pd.Series(["small"], name="sml")),
    ],
    ids=["startswith", "endswith", "re_search", "lstrip", "rstrip", "strip"],
)
def test_enum_as_string(enum_t, expr_fn, expected):
    expr = expr_fn(enum_t.sml).name("sml")
    res = expr.execute()
    tm.assert_series_equal(res, expected)


def test_builtin_scalar_udf(con):
    @udf.scalar.builtin
    def soundex(a: str) -> str:
        """Soundex of a string."""

    expr = soundex("foo")
    result = con.execute(expr)
    assert result == "F000"


def test_builtin_agg_udf(con):
    @udf.agg.builtin
    def json_arrayagg(a) -> str:
        """Glom together some JSON."""

    ft = con.tables.functional_alltypes[:5]
    expr = json_arrayagg(ft.string_col)
    result = expr.execute()
    expected = json.dumps(list(map(str, range(5))), separators=",:")
    assert result == expected
