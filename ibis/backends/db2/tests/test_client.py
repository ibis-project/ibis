from __future__ import annotations

import decimal

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
    con.con.commit()

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


def test_execute_keeps_decimal_exact(con):
    import pandas as pd

    values = [
        decimal.Decimal("123456789012345678.99"),
        decimal.Decimal("-0.01"),
        None,
    ]
    name = gen_name("db2_test")
    con.create_table(name, schema={"d": "decimal(20, 2)"})
    try:
        con.insert(name, pd.DataFrame({"d": values}))
        result = con.table(name).execute()["d"]
        assert sorted(result.dropna()) == sorted(v for v in values if v is not None)
        assert result.isna().sum() == 1
    finally:
        con.drop_table(name, force=True)


def test_execute_decimal_wider_than_float(con):
    # 25 nines do not fit in a float64; they used to raise TypeError.
    value = decimal.Decimal("9" * 25 + "." + "9" * 6)
    result = con.execute(ibis.literal(value, type="decimal(31, 6)"))
    assert result == value


@pytest.mark.parametrize(
    ("key", "override"),
    [
        ("PWD", {"password": "secret;junk"}),
        ("UID", {"user": "someone;else"}),
        ("HOSTNAME", {"host": "localhost;x"}),
        ("DATABASE", {"database": "db;x"}),
        ("CURRENTSCHEMA", {"schema": "s;x"}),
        ("CONNECTTIMEOUT", {"connecttimeout": "5;x"}),
    ],
)
def test_connect_rejects_semicolon(key, override):
    # Without the check the driver ends the value at `;`, so `secret;junk`
    # would be accepted as `secret`.
    kwargs = {
        "database": DB2_DATABASE,
        "host": DB2_HOST,
        "port": DB2_PORT,
        "user": DB2_USER,
        "password": DB2_PASS,
        **override,
    }
    with pytest.raises(ValueError, match=key):
        ibis.db2.connect(**kwargs)


@pytest.mark.parametrize(
    ("make_expr", "expected"),
    [
        pytest.param(lambda t: t.x / t.x.sum(), [0.1, 0.2, 0.3, 0.4], id="sum"),
        pytest.param(lambda t: t.x / t.count(), [0.25, 0.5, 0.75, 1.0], id="count"),
        pytest.param(lambda t: t.x / t.y, [0.5] * 4, id="column"),
        pytest.param(lambda t: t.x / 2, [0.5, 1.0, 1.5, 2.0], id="literal"),
        pytest.param(lambda t: t.x / t.x, [1.0] * 4, id="self"),
    ],
)
def test_divide_integers_is_not_integer_division(con, make_expr, expected):
    # Used to return whole numbers (0.0 for `x / x.sum()`): the integer-to-DOUBLE
    # cast that sqlglot adds was lost when `compile` generated SQL without copying.
    t = ibis.memtable({"x": [1, 2, 3, 4], "y": [2, 4, 6, 8]})
    expr = t.mutate(r=make_expr(t)).order_by("x")
    assert con.execute(expr)["r"].tolist() == pytest.approx(expected)


def test_divide_integers_after_filter(con):
    t = ibis.memtable({"x": [1, 2, 3, 4]})
    filtered = t.filter(t.x > 1)
    expr = filtered.mutate(r=filtered.x / filtered.x.sum()).order_by("x")
    assert con.execute(expr)["r"].tolist() == pytest.approx([2 / 9, 3 / 9, 4 / 9])


def test_divide_decimal_and_float_are_unchanged(con):
    t = ibis.memtable(
        {"f": [1.0, 3.0], "d": [decimal.Decimal("1.50"), decimal.Decimal("3.50")]},
        schema={"f": "float64", "d": "decimal(10, 2)"},
    )
    expr = t.mutate(rf=t.f / t.f.sum(), rd=t.d / t.d.sum()).order_by("f")
    result = con.execute(expr)
    assert result["rf"].tolist() == pytest.approx([0.25, 0.75])
    assert [float(v) for v in result["rd"]] == pytest.approx([0.3, 0.7], abs=1e-3)
