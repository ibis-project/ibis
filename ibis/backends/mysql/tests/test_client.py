from __future__ import annotations

import json
from datetime import date
from operator import methodcaller

import pandas as pd
import pandas.testing as tm
import pytest
import sqlalchemy as sa
from packaging.version import parse as vparse
from pytest import param
from sqlalchemy.dialects import mysql

import ibis
import ibis.expr.datatypes as dt
from ibis import udf
from ibis.backends.base.sql.alchemy.geospatial import geospatial_supported
from ibis.util import gen_name

if geospatial_supported:
    import geoalchemy2
else:
    geoalchemy2 = None

MYSQL_TYPES = [
    param("tinyint", dt.int8, id="tinyint"),
    param("int1", dt.int8, id="int1"),
    param("boolean", dt.int8, id="boolean"),
    param("smallint", dt.int16, id="smallint"),
    param("int2", dt.int16, id="int2"),
    # ("mediumint", dt.int32), => https://github.com/tobymao/sqlglot/issues/2109
    # ("int3", dt.int32), => https://github.com/tobymao/sqlglot/issues/2109
    param("int", dt.int32, id="int"),
    param("int4", dt.int32, id="int4"),
    param("integer", dt.int32, id="integer"),
    param("bigint", dt.int64, id="bigint"),
    param("decimal", dt.Decimal(10, 0), id="decimal"),
    param("decimal(5, 2)", dt.Decimal(5, 2), id="decimal_5_2"),
    param("dec", dt.Decimal(10, 0), id="dec"),
    param("numeric", dt.Decimal(10, 0), id="numeric"),
    param("fixed", dt.Decimal(10, 0), id="fixed"),
    param("float", dt.float32, id="float"),
    param("double", dt.float64, id="double"),
    param("timestamp", dt.Timestamp("UTC"), id="timestamp"),
    param("date", dt.date, id="date"),
    param("time", dt.time, id="time"),
    param("datetime", dt.timestamp, id="datetime"),
    param("year", dt.int8, id="year"),
    param("char(32)", dt.string, id="char"),
    param("char byte", dt.binary, id="char_byte"),
    param("varchar(42)", dt.string, id="varchar"),
    param("mediumtext", dt.string, id="mediumtext"),
    param("text", dt.string, id="text"),
    param("binary(42)", dt.binary, id="binary"),
    param("varbinary(42)", dt.binary, id="varbinary"),
    param("bit(1)", dt.int8, id="bit_1"),
    param("bit(9)", dt.int16, id="bit_9"),
    param("bit(17)", dt.int32, id="bit_17"),
    param("bit(33)", dt.int64, id="bit_33"),
    # mariadb doesn't have a distinct json type
    param("json", dt.string, id="json"),
    param("enum('small', 'medium', 'large')", dt.string, id="enum"),
    # con.table(name) first parses the type correctly as ibis inet using sqlglot,
    # then convert these types to sqlalchemy types then a sqlalchemy table to
    # get the ibis schema again from the alchemy types, but alchemy doesn't
    # support inet6 so it gets converted to string eventually
    # ("inet6", dt.inet),
    param("set('a', 'b', 'c', 'd')", dt.Array(dt.string), id="set"),
    param("mediumblob", dt.binary, id="mediumblob"),
    param("blob", dt.binary, id="blob"),
    param(
        "uuid",
        dt.uuid,
        marks=[
            pytest.mark.xfail(
                condition=vparse(sa.__version__) < vparse("2"),
                reason="geoalchemy2 0.14.x doesn't work",
            )
        ],
        id="uuid",
    ),
]


@pytest.mark.parametrize(("mysql_type", "expected_type"), MYSQL_TYPES)
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
@pytest.mark.xfail(
    geospatial_supported and vparse(geoalchemy2.__version__) > vparse("0.13.3"),
    reason="geoalchemy2 issues when using 0.14.x",
    raises=sa.exc.OperationalError,
)
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
