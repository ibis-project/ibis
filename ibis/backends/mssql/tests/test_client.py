from __future__ import annotations

from urllib.parse import urlencode

import pytest
import sqlglot as sg
import sqlglot.expressions as sge
from pytest import param

import ibis
import ibis.common.exceptions as com
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
from ibis.backends.tests.errors import PyODBCProgrammingError
from ibis.util import gen_name

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
    # Character strings
    ("CHAR", dt.String(length=1)),
    ("VARCHAR", dt.String(length=1)),
    ("CHAR(73)", dt.String(length=73)),
    ("VARCHAR(73)", dt.String(length=73)),
    # Unicode character strings
    ("NCHAR", dt.String(length=1)),
    ("NVARCHAR", dt.String(length=1)),
    ("NCHAR(42)", dt.String(length=42)),
    ("NVARCHAR(42)", dt.String(length=42)),
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
    assert expr.execute() == ft.filter(ft.id == 1).count().execute()


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


def test_list_tables(con):
    assert set(con.list_tables()) >= {
        "astronauts",
        "awards_players",
        "batting",
        "diamonds",
        "functional_alltypes",
        "win",
    }

    restore_tables = ["restorefile", "restorefilegroup", "restorehistory"]

    assert con.list_tables(database="msdb.dbo", like="restore") == restore_tables
    assert con.list_tables(database=("msdb", "dbo"), like="restore") == restore_tables


def test_create_temp_table_from_obj(con):
    obj = {"team": ["john", "joe"]}

    t = con.create_table("team", obj, temp=True)

    try:
        t2 = con.table("##team", database="tempdb.dbo")

        assert t.to_pyarrow().equals(t2.to_pyarrow())

        persisted_from_temp = con.create_table("fuhreal", t2)

        try:
            assert "fuhreal" in con.list_tables()
            assert persisted_from_temp.to_pyarrow().equals(t2.to_pyarrow())
        finally:
            con.drop_table("fuhreal")
    finally:
        con.drop_table("#team", force=True)


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


def test_dot_sql_with_unnamed_columns(con):
    expr = con.sql(
        "SELECT CAST('2024-01-01 00:00:00' AS DATETIMEOFFSET), 'a' + 'b', 1 AS [col42]"
    )

    schema = expr.schema()
    names = schema.names

    assert len(names) == 3

    assert names[0].startswith("ibis_col")
    assert names[1].startswith("ibis_col")
    assert names[2] == "col42"

    assert schema.types == (
        dt.Timestamp(timezone="UTC", scale=7),
        dt.String(nullable=False, length=2),
        dt.Int32(nullable=False),
    )

    df = expr.execute()
    assert len(df) == 1


def test_dot_sql_error_handling(con):
    with pytest.raises(com.IbisInputError, match="Invalid column name"):
        con.sql("SELECT not_a_column")


@pytest.mark.parametrize(
    "temp",
    [
        param(
            True,
            marks=pytest.mark.xfail(
                raises=PyODBCProgrammingError,
                reason="dropping temp tables isn't implemented",
            ),
        ),
        False,
        None,
    ],
    ids=[
        "temp",
        "no-temp",
        "no-temp-none",
    ],
)
def test_create_temp_table(con, temp):
    t = con.create_table(
        name := gen_name("mssql_delete_me"),
        schema={"a": "int"},
        temp=temp,
    )
    try:
        assert int(t.count().execute()) == 0
        assert t.schema() == ibis.schema({"a": "int"})
        assert t.columns == ("a",)
    finally:
        con.drop_table(name)


def test_escape_special_characters():
    test_func = ibis.backends.mssql.Backend._escape_special_characters
    assert test_func("1bis_Testing!") == "{1bis_Testing!}"
    assert test_func("{1bis_Testing!") == "{{1bis_Testing!}"
    assert test_func("1bis_Testing!}") == "{1bis_Testing!}}}"
    assert test_func("{1bis_Testing!}") == "{{1bis_Testing!}}}"
    assert test_func("1bis}Testing!") == "{1bis}}Testing!}"
    assert test_func("{R;3G1/8Al2AniRye") == "{{R;3G1/8Al2AniRye}"
    assert test_func("{R;3G1/8Al2AniRye}") == "{{R;3G1/8Al2AniRye}}}"


def test_non_ascii_column_name(con):
    expr = con.sql("SELECT 1 AS [калона]")
    schema = expr.schema()
    names = schema.names
    assert len(names) == 1
    assert names[0] == "калона"


def test_mssql_without_password_is_valid():
    with pytest.raises(
        PyODBCProgrammingError, match=f"Login failed for user '{MSSQL_USER}'"
    ):
        ibis.mssql.connect(
            user=MSSQL_USER,
            host=MSSQL_HOST,
            password=None,
            port=MSSQL_PORT,
            database=IBIS_TEST_MSSQL_DB,
            driver=MSSQL_PYODBC_DRIVER,
        )


@pytest.mark.parametrize(
    "database", ["ibis-testing.dbo", ("ibis-testing", "dbo")], ids=["string", "tuple"]
)
def test_list_tables_with_dash(con, database):
    assert con.list_tables(database=database)


def test_rank_no_window_frame(snapshot):
    t = ibis.table(schema=dict(color=str, price=int), name="diamonds_sample")
    expr = t.mutate(ibis.rank().over(group_by="color", order_by="price"))
    sql = ibis.to_sql(expr, dialect="mssql")

    snapshot.assert_match(sql, "out.sql")
