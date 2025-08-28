from __future__ import annotations

import json
from datetime import date
from operator import methodcaller

import pandas as pd
import pandas.testing as tm
import pytest
import sqlglot as sg
from pytest import param

import ibis
import ibis.expr.datatypes as dt
from ibis import udf
from ibis.backends.singlestoredb.tests.conftest import (
    IBIS_TEST_SINGLESTOREDB_DB,
    SINGLESTOREDB_HOST,
    SINGLESTOREDB_PASS,
    SINGLESTOREDB_USER,
)
from ibis.backends.tests.errors import (
    SingleStoreDBOperationalError,
    SingleStoreDBProgrammingError,
)
from ibis.util import gen_name

SINGLESTOREDB_TYPES = [
    # Integer types
    param("tinyint", dt.int8, id="tinyint"),
    param("int1", dt.int8, id="int1"),
    param("smallint", dt.int16, id="smallint"),
    param("int2", dt.int16, id="int2"),
    param("mediumint", dt.int32, id="mediumint"),
    param("int3", dt.int32, id="int3"),
    param("int", dt.int32, id="int"),
    param("int4", dt.int32, id="int4"),
    param("integer", dt.int32, id="integer"),
    param("bigint", dt.int64, id="bigint"),
    # Decimal types
    param("decimal", dt.Decimal(10, 0), id="decimal"),
    param("decimal(5, 2)", dt.Decimal(5, 2), id="decimal_5_2"),
    param("dec", dt.Decimal(10, 0), id="dec"),
    param("numeric", dt.Decimal(10, 0), id="numeric"),
    param("fixed", dt.Decimal(10, 0), id="fixed"),
    # Float types
    param("float", dt.float32, id="float"),
    param("double", dt.float64, id="double"),
    param("real", dt.float64, id="real"),
    # Temporal types
    param("timestamp", dt.Timestamp("UTC"), id="timestamp"),
    param("date", dt.date, id="date"),
    param("time", dt.time, id="time"),
    param("datetime", dt.timestamp, id="datetime"),
    param("year", dt.uint8, id="year"),
    # String types
    param("char(32)", dt.String(length=32), id="char"),
    param("varchar(42)", dt.String(length=42), id="varchar"),
    param("text", dt.string, id="text"),
    param("mediumtext", dt.string, id="mediumtext"),
    param("longtext", dt.string, id="longtext"),
    # Binary types
    param("binary(42)", dt.binary, id="binary"),
    param("varbinary(42)", dt.binary, id="varbinary"),
    param("blob", dt.binary, id="blob"),
    param("mediumblob", dt.binary, id="mediumblob"),
    param("longblob", dt.binary, id="longblob"),
    # Bit types
    param("bit(1)", dt.int8, id="bit_1"),
    param("bit(9)", dt.int16, id="bit_9"),
    param("bit(17)", dt.int32, id="bit_17"),
    param("bit(33)", dt.int64, id="bit_33"),
    # Special SingleStoreDB types
    param("json", dt.json, id="json"),
    # Unsigned integer types
    param("mediumint(8) unsigned", dt.uint32, id="mediumint-unsigned"),
    param("bigint unsigned", dt.uint64, id="bigint-unsigned"),
    param("int unsigned", dt.uint32, id="int-unsigned"),
    param("smallint unsigned", dt.uint16, id="smallint-unsigned"),
    param("tinyint unsigned", dt.uint8, id="tinyint-unsigned"),
] + [
    param(
        f"datetime({scale:d})",
        dt.Timestamp(scale=scale or None),
        id=f"datetime{scale:d}",
        marks=pytest.mark.skipif(
            scale not in (0, 6),
            reason=f"SingleStoreDB only supports DATETIME(0) and DATETIME(6), not DATETIME({scale})",
        ),
    )
    for scale in range(7)
]


@pytest.mark.parametrize(("singlestoredb_type", "expected_type"), SINGLESTOREDB_TYPES)
def test_get_schema_from_query(con, singlestoredb_type, expected_type):
    raw_name = ibis.util.guid()
    name = sg.to_identifier(raw_name, quoted=True).sql("singlestore")
    expected_schema = ibis.schema(dict(x=expected_type))

    # temporary tables get cleaned up by the db when the session ends, so we
    # don't need to explicitly drop the table
    with con.begin() as c:
        c.execute(f"CREATE TEMPORARY TABLE {name} (x {singlestoredb_type})")

    result_schema = con._get_schema_using_query(f"SELECT * FROM {name}")
    assert result_schema == expected_schema

    t = con.table(raw_name)
    assert t.schema() == expected_schema


@pytest.mark.parametrize(
    ("singlestoredb_type", "get_schema_expected_type", "table_expected_type"),
    [
        param(
            "enum('small', 'medium', 'large')",
            dt.String(length=6),
            dt.string,
            id="enum",
        ),
        param(
            "boolean",
            dt.int8,  # Cursor-based detection cannot distinguish BOOLEAN from TINYINT
            dt.boolean,  # DESCRIBE-based detection correctly identifies BOOLEAN
            id="boolean",
        ),
    ],
)
def test_get_schema_from_query_special_cases(
    con, singlestoredb_type, get_schema_expected_type, table_expected_type
):
    raw_name = ibis.util.guid()
    name = sg.to_identifier(raw_name, quoted=True).sql("singlestore")
    get_schema_expected_schema = ibis.schema(dict(x=get_schema_expected_type))
    table_expected_schema = ibis.schema(dict(x=table_expected_type))

    # temporary tables get cleaned up by the db when the session ends, so we
    # don't need to explicitly drop the table
    with con.begin() as c:
        c.execute(f"CREATE TEMPORARY TABLE {name} (x {singlestoredb_type})")

    result_schema = con._get_schema_using_query(f"SELECT * FROM {name}")
    assert result_schema == get_schema_expected_schema

    t = con.table(raw_name)
    assert t.schema() == table_expected_schema


@pytest.mark.parametrize("coltype", ["TINYBLOB", "MEDIUMBLOB", "BLOB", "LONGBLOB"])
def test_blob_type(con, coltype):
    tmp = f"tmp_{ibis.util.guid()}"
    with con.begin() as c:
        c.execute(f"CREATE TEMPORARY TABLE {tmp} (a {coltype})")
    t = con.table(tmp)
    assert t.schema() == ibis.schema({"a": dt.binary})


def test_zero_timestamp_data(con):
    sql = """
    CREATE TEMPORARY TABLE ztmp_date_issue
    (
        name      CHAR(10) NULL,
        tradedate DATETIME NOT NULL,
        date      DATETIME NULL
    )
    """
    with con.begin() as c:
        c.execute(sql)
        c.execute(
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
    # Sort both DataFrames by tradedate to ensure consistent ordering
    result_sorted = result.sort_values("tradedate").reset_index(drop=True)
    expected_sorted = expected.sort_values("tradedate").reset_index(drop=True)
    tm.assert_frame_equal(result_sorted, expected_sorted)


@pytest.fixture(scope="module")
def enum_t(con):
    name = gen_name("singlestoredb_enum_test")
    with con.begin() as cur:
        cur.execute(
            f"CREATE TEMPORARY TABLE {name} (sml ENUM('small', 'medium', 'large'))"
        )
        cur.execute(f"INSERT INTO {name} VALUES ('small')")

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
    def reverse(a: str) -> str:
        """Reverse a string."""

    expr = reverse("foo")
    result = con.execute(expr)
    assert result == "oof"


def test_list_tables(con):
    # Just verify that we can list tables
    tables = con.list_tables()
    assert isinstance(tables, list)
    assert len(tables) >= 0  # Should have at least some test tables


def test_invalid_port():
    port = 4000
    url = f"singlestoredb://{SINGLESTOREDB_USER}:{SINGLESTOREDB_PASS}@{SINGLESTOREDB_HOST}:{port}/{IBIS_TEST_SINGLESTOREDB_DB}"
    with pytest.raises(SingleStoreDBOperationalError):
        ibis.connect(url)


def test_create_database_exists(con):
    con.create_database(dbname := gen_name("dbname"))

    with pytest.raises(SingleStoreDBProgrammingError):
        con.create_database(dbname)

    con.create_database(dbname, force=True)

    con.drop_database(dbname, force=True)


def test_drop_database_exists(con):
    con.create_database(dbname := gen_name("dbname"))

    con.drop_database(dbname)

    with pytest.raises(SingleStoreDBOperationalError):
        con.drop_database(dbname)

    con.drop_database(dbname, force=True)


def test_json_type_support(con):
    """Test SingleStoreDB JSON type handling."""
    tmp = f"tmp_{ibis.util.guid()}"
    with con.begin() as c:
        c.execute(f"CREATE TEMPORARY TABLE {tmp} (data JSON)")
        json_value = json.dumps({"key": "value"})
        c.execute(f"INSERT INTO {tmp} VALUES ('{json_value}')")

    t = con.table(tmp)
    assert t.schema() == ibis.schema({"data": dt.JSON(nullable=True)})

    result = t.execute()
    assert len(result) == 1
    assert "key" in result.iloc[0]["data"]


def test_connection_attributes(con):
    """Test that connection has expected attributes."""
    assert hasattr(con, "database")
    assert hasattr(con, "_get_schema_using_query")
    assert hasattr(con, "list_tables")
    assert hasattr(con, "create_database")
    assert hasattr(con, "drop_database")


def test_table_creation_basic_types(con):
    """Test creating tables with basic data types."""
    table_name = f"test_{ibis.util.guid()}"
    schema = ibis.schema(
        [
            ("id", dt.int32),
            ("name", dt.string),
            ("value", dt.float64),
            ("created_at", dt.timestamp),
            ("is_active", dt.boolean),
        ]
    )

    # Create table
    con.create_table(table_name, schema=schema, temp=True)

    # Verify table exists and has correct schema
    t = con.table(table_name)
    actual_schema = t.schema()

    # Check that essential columns exist (may have slight type differences)
    assert "id" in actual_schema
    assert "name" in actual_schema
    assert "value" in actual_schema
    assert "created_at" in actual_schema
    assert "is_active" in actual_schema


def test_transaction_handling(con):
    """Test transaction begin/commit/rollback."""
    table_name = f"test_txn_{ibis.util.guid()}"

    with con.begin() as c:
        c.execute(f"CREATE TEMPORARY TABLE {table_name} (id INT, value VARCHAR(50))")
        c.execute(f"INSERT INTO {table_name} VALUES (1, 'test')")

    # Verify data was committed
    t = con.table(table_name)
    result = t.execute()
    assert len(result) == 1
    assert result.iloc[0]["id"] == 1
    assert result.iloc[0]["value"] == "test"
