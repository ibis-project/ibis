from __future__ import annotations

from urllib.parse import urlencode

import pytest
import sqlglot as sg
import sqlglot.expressions as sge
from pytest import param

import ibis
import ibis.expr.datatypes as dt
from ibis import udf
from ibis.backends.mssql.tests.conftest import (
    IBIS_TEST_MSSQL_DB,
    MSSQL_HOST,
    MSSQL_PASS,
    MSSQL_PORT,
    MSSQL_PYODBC_DRIVER,
    MSSQL_USER,
)

RAW_DB_TYPES = [
    # Exact numbers
    ("BIGINT", dt.int64),
    ("BIT", dt.boolean),
    ("DECIMAL", dt.Decimal(precision=18, scale=0)),
    ("DECIMAL(5, 2)", dt.Decimal(precision=5, scale=2)),
    ("INT", dt.int32),
    ("MONEY", dt.Decimal(19, 4)),
    ("NUMERIC", dt.Decimal(18, 0)),
    ("NUMERIC(10,5)", dt.Decimal(10, 5)),
    ("NUMERIC(14,3)", dt.Decimal(14, 3)),
    ("SMALLINT", dt.int16),
    ("SMALLMONEY", dt.Decimal(10, 4)),
    ("TINYINT", dt.uint8),
    # Approximate numerics
    ("REAL", dt.float32),
    ("FLOAT", dt.float64),
    ("FLOAT(3)", dt.float32),
    ("FLOAT(25)", dt.float64),
    ("FLOAT(37)", dt.float64),
    # Date and time
    ("DATE", dt.date),
    ("TIME", dt.time),
    ("DATETIME2", dt.timestamp(scale=7)),
    ("DATETIMEOFFSET", dt.timestamp(scale=7, timezone="UTC")),
    ("SMALLDATETIME", dt.Timestamp(scale=0)),
    ("DATETIME", dt.Timestamp(scale=3)),
    # Characters strings
    ("CHAR", dt.string),
    ("VARCHAR", dt.string),
    # Unicode character strings
    ("NCHAR", dt.string),
    ("NVARCHAR", dt.string),
    # Binary strings
    ("BINARY", dt.binary),
    ("VARBINARY", dt.binary),
    ("IMAGE", dt.binary),
    # Other data types
    ("UNIQUEIDENTIFIER", dt.uuid),
    ("TIMESTAMP", dt.binary(nullable=False)),
    ("DATETIME2(4)", dt.timestamp(scale=4)),
    ("DATETIMEOFFSET(5)", dt.timestamp(scale=5, timezone="UTC")),
    ("GEOMETRY", dt.geometry),
    ("GEOGRAPHY", dt.geography),
    ("HIERARCHYID", dt.string),
]
PARAM_TYPES = [
    param(
        "TEXT",
        dt.string,
        marks=pytest.mark.notyet(
            ["mssql"], reason="Not supported by UTF-8 aware collations"
        ),
    ),
    param(
        "NTEXT",
        dt.string,
        marks=pytest.mark.notyet(
            ["mssql"], reason="Not supported by UTF-8 aware collations"
        ),
    ),
]
DB_TYPES = RAW_DB_TYPES + PARAM_TYPES


@pytest.mark.parametrize(("server_type", "expected_type"), DB_TYPES, ids=str)
def test_get_schema(con, server_type, expected_type, temp_table):
    with con.begin() as c:
        c.execute(f"CREATE TABLE [{temp_table}] (x {server_type})")

    expected_schema = ibis.schema(dict(x=expected_type))

    assert con.get_schema(temp_table) == expected_schema
    assert con.table(temp_table).schema() == expected_schema
    assert con.sql(f"SELECT * FROM [{temp_table}]").schema() == expected_schema


def test_schema_type_order(con, temp_table):
    columns = []
    pairs = {}

    quoted = con.compiler.quoted
    dialect = con.dialect
    table_id = sg.to_identifier(temp_table, quoted=quoted)

    for i, (server_type, expected_type) in enumerate(RAW_DB_TYPES):
        column_name = f"col_{i}"
        columns.append(
            sge.ColumnDef(
                this=sg.to_identifier(column_name, quoted=quoted), kind=server_type
            )
        )
        pairs[column_name] = expected_type

    query = sge.Create(
        kind="TABLE", this=sge.Schema(this=table_id, expressions=columns)
    )
    stmt = query.sql(dialect)

    with con.begin() as c:
        c.execute(stmt)

    expected_schema = ibis.schema(pairs)

    assert con.get_schema(temp_table) == expected_schema
    assert con.table(temp_table).schema() == expected_schema

    raw_sql = sg.select("*").from_(table_id).sql(dialect)
    assert con.sql(raw_sql).schema() == expected_schema


def test_builtin_scalar_udf(con):
    @udf.scalar.builtin
    def difference(a: str, b: str) -> int:
        """Soundex difference between two strings."""

    expr = difference("foo", "moo")
    result = con.execute(expr)
    assert result == 3


def test_builtin_agg_udf(con):
    @udf.agg.builtin
    def count_big(x) -> int:
        """The biggest of counts."""

    ft = con.tables.functional_alltypes
    expr = count_big(ft.id)
    assert expr.execute() == ft.count().execute()


def test_builtin_agg_udf_filtered(con):
    @udf.agg.builtin
    def count_big(x, where: bool = True) -> int:
        """The biggest of counts."""

    ft = con.tables.functional_alltypes
    expr = count_big(ft.id)

    expr = count_big(ft.id, where=ft.id == 1)
    assert expr.execute() == ft[ft.id == 1].count().execute()


@pytest.mark.parametrize("string", ["a", " ", "a ", " a", ""])
def test_glorious_length_function_hack(con, string):
    """Test that the length function works as expected.

    Why wouldn't it, you ask?

    https://learn.microsoft.com/en-us/sql/t-sql/functions/len-transact-sql?view=sql-server-ver16#remarks
    """
    lit = ibis.literal(string)
    expr = lit.length()
    result = con.execute(expr)
    assert result == len(string)


def test_list_tables_schema_warning_refactor(con):
    assert set(con.list_tables()) >= {
        "astronauts",
        "awards_players",
        "batting",
        "diamonds",
        "functional_alltypes",
        "win",
    }

    restore_tables = ["restorefile", "restorefilegroup", "restorehistory"]

    with pytest.warns(FutureWarning):
        assert (
            con.list_tables(database="msdb", schema="dbo", like="restore")
            == restore_tables
        )

    assert con.list_tables(database="msdb.dbo", like="restore") == restore_tables
    assert con.list_tables(database=("msdb", "dbo"), like="restore") == restore_tables


def test_create_temp_table_from_obj(con):
    obj = {"team": ["john", "joe"]}

    t = con.create_table("team", obj, temp=True)

    t2 = con.table("##team", database="tempdb.dbo")

    assert t.to_pyarrow().equals(t2.to_pyarrow())

    persisted_from_temp = con.create_table("fuhreal", t2)

    assert "fuhreal" in con.list_tables()

    assert persisted_from_temp.to_pyarrow().equals(t2.to_pyarrow())

    con.drop_table("fuhreal")


@pytest.mark.parametrize("explicit_schema", [False, True])
def test_create_temp_table_from_expression(con, explicit_schema, temp_table):
    t = ibis.memtable(
        {"x": [1, 2, 3], "y": ["a", "b", "c"]}, schema={"x": "int64", "y": "str"}
    )
    t2 = con.create_table(
        temp_table, t, temp=True, schema=t.schema() if explicit_schema else None
    )
    res = con.to_pandas(t.order_by("y"))
    sol = con.to_pandas(t2.order_by("y"))
    assert res.equals(sol)


def test_from_url():
    user = MSSQL_USER
    password = MSSQL_PASS
    host = MSSQL_HOST
    port = MSSQL_PORT
    database = IBIS_TEST_MSSQL_DB
    driver = MSSQL_PYODBC_DRIVER
    new_con = ibis.connect(
        f"mssql://{user}:{password}@{host}:{port}/{database}?{urlencode(dict(driver=driver))}"
    )
    result = new_con.sql("SELECT 1 AS [a]").to_pandas().a.iat[0]
    assert result == 1
