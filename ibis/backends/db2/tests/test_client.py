from __future__ import annotations

import pytest

import ibis
import ibis.expr.datatypes as dt
from ibis.backends.db2.tests.conftest import (
    DB2_DATABASE,
    DB2_HOST,
    DB2_PASS,
    DB2_PORT,
    DB2_USER,
)
from ibis.util import gen_name


def test_connect_url():
    new_con = ibis.connect(
        f"db2://{DB2_USER}:{DB2_PASS}@{DB2_HOST}:{DB2_PORT}/{DB2_DATABASE}"
    )
    result = new_con.raw_sql("SELECT 1 FROM SYSIBM.SYSDUMMY1").fetchone()[0]
    assert result == 1


def test_list_tables(con):
    tables = con.list_tables()
    assert set(tables) >= {
        "astronauts",
        "awards_players",
        "batting",
        "diamonds",
        "functional_alltypes",
        "win",
    }


def test_current_database(con):
    db = con.current_database
    assert isinstance(db, str)
    assert len(db) > 0


def test_version(con):
    v = con.version
    assert isinstance(v, str)
    assert len(v) > 0


@pytest.mark.parametrize(
    ("server_type", "expected_type"),
    [
        ("SMALLINT", dt.int16),
        ("INTEGER", dt.int32),
        ("BIGINT", dt.int64),
        ("REAL", dt.float32),
        ("DOUBLE", dt.float64),
        ("DATE", dt.date),
        ("TIME", dt.time),
        ("TIMESTAMP", dt.timestamp),
        ("VARCHAR(100)", dt.string),
        ("CHAR(10)", dt.string),
        ("CLOB", dt.string),
    ],
)
def test_get_schema_types(con, server_type, expected_type, temp_table):
    with con._safe_raw_sql(f'CREATE TABLE "{temp_table}" (x {server_type})'):
        pass
    con._connection.commit()

    try:
        schema = con.get_schema(temp_table)
        # DB2 uppercases unquoted column names in SYSCAT.COLUMNS ("x" -> "X")
        key = "x" if "x" in schema else "X"
        assert isinstance(schema[key], type(expected_type))
    finally:
        con.drop_table(temp_table, force=True)


def test_get_schema(con):
    schema = con.get_schema("functional_alltypes")
    assert "id" in schema
    assert "bool_col" in schema
    assert "string_col" in schema
    assert "timestamp_col" in schema


def test_create_and_drop_table(con):
    name = gen_name("db2_test")
    t = con.create_table(name, schema={"a": "int32", "b": "string"})
    try:
        assert len(t.execute()) == 0
        assert set(t.columns) == {"a", "b"}
    finally:
        con.drop_table(name, force=True)


def test_drop_nonexistent_table_force(con):
    # should not raise
    con.drop_table(gen_name("db2_nonexistent"), force=True)


def test_insert_and_select(con):
    import pandas as pd

    name = gen_name("db2_test")
    con.create_table(name, schema={"x": "int64", "y": "string"})
    try:
        df = pd.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})
        con.insert(name, df)
        t = con.table(name)
        result = t.order_by("x").execute()
        assert len(result) == 3
        assert list(result["y"]) == ["a", "b", "c"]
    finally:
        con.drop_table(name, force=True)


def test_overwrite_insert(con):
    import pandas as pd

    name = gen_name("db2_test")
    df1 = pd.DataFrame({"x": [1, 2, 3]})
    df2 = pd.DataFrame({"x": [10, 20]})
    con.create_table(name, df1)
    try:
        con.insert(name, df2, overwrite=True)
        t = con.table(name)
        result = t.execute()
        assert len(result) == 2
        assert sorted(result["x"].tolist()) == [10, 20]
    finally:
        con.drop_table(name, force=True)
