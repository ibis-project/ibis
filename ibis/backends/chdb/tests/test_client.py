"""Connection, DDL and catalog behavior of the chDB backend."""

from __future__ import annotations

import sys

import pandas as pd
import pandas.testing as tm
import pytest

import ibis
import ibis.common.exceptions as com
from ibis.backends.chdb import _import_chdb


def test_connect_and_version(mem):
    assert mem.name == "chdb"
    assert isinstance(mem.version, str) and mem.version


def test_raw_sql(mem):
    res = mem.raw_sql("SELECT 1 AS a, 'x' AS b", fmt="CSV")
    assert str(res).strip() == '1,"x"'


def test_create_and_drop_table(mem):
    t = ibis.memtable({"x": [1, 2, 3]})
    mem.create_table("t1", obj=t, engine="Memory")
    assert "t1" in mem.list_tables()
    mem.drop_table("t1")
    assert "t1" not in mem.list_tables()


def test_create_table_from_schema(mem):
    mem.create_table(
        "t2", schema=ibis.schema({"a": "int64", "s": "string"}), engine="Memory"
    )
    assert dict(mem.get_schema("t2")) == dict(
        ibis.schema({"a": "int64", "s": "string"})
    )


def test_create_table_overwrite(mem):
    mem.create_table("t3", obj=ibis.memtable({"x": [1]}), engine="Memory")
    mem.create_table(
        "t3", obj=ibis.memtable({"x": [1, 2, 3]}), engine="Memory", overwrite=True
    )
    assert mem.table("t3").count().execute() == 3


def test_list_tables_like(mem):
    mem.create_table("apple", obj=ibis.memtable({"x": [1]}), engine="Memory")
    mem.create_table("apricot", obj=ibis.memtable({"x": [1]}), engine="Memory")
    mem.create_table("banana", obj=ibis.memtable({"x": [1]}), engine="Memory")
    assert set(mem.list_tables(like="ap")) == {"apple", "apricot"}


def test_get_schema(mem):
    mem.create_table(
        "typed",
        schema=ibis.schema({"i": "int32", "f": "float64", "s": "string"}),
        engine="Memory",
    )
    assert dict(mem.get_schema("typed")) == dict(
        ibis.schema({"i": "int32", "f": "float64", "s": "string"})
    )


def test_create_and_drop_database(mem):
    mem.create_database("scratch", force=True)
    assert "scratch" in mem.list_databases()
    mem.drop_database("scratch", force=True)
    assert "scratch" not in mem.list_databases()


def test_create_view(mem):
    mem.create_table("base", obj=ibis.memtable({"x": [1, 2, 3, 4]}), engine="Memory")
    mem.create_view("v", mem.table("base").filter(ibis._.x > 2))
    assert mem.table("v").count().execute() == 2


def test_table_roundtrip_values(mem):
    df = pd.DataFrame({"x": [1, 2, 3], "s": ["a", "b", "c"]})
    mem.create_table("rt", obj=ibis.memtable(df), engine="Memory")
    got = mem.table("rt").order_by("x").execute()
    tm.assert_frame_equal(got.reset_index(drop=True), df)


def test_multiple_connections_same_path(chdb_path):
    c1 = ibis.chdb.connect(chdb_path)
    c1.create_table("shared", obj=ibis.memtable({"a": [1, 2]}), engine="Memory")
    # chDB permits a second connection against the same path.
    c2 = ibis.chdb.connect(chdb_path)
    assert c2.name == "chdb"
    c1.disconnect()
    c2.disconnect()


def test_connect_via_database_kwarg(chdb_path):
    con = ibis.chdb.connect(database=chdb_path)
    try:
        assert con.name == "chdb"
        assert con.execute(ibis.memtable({"x": [1, 2, 3]}).x.sum()) == 6
    finally:
        con.disconnect()


def test_connect_via_url(chdb_path):
    # chdb://<path> must resolve to a path-based connection (not the inherited
    # ClickHouse host/port URL parser).
    con = ibis.connect(f"chdb://{chdb_path}")
    try:
        assert con.name == "chdb"
        con.create_table("u", obj=ibis.memtable({"x": [1, 2]}), engine="Memory")
        assert "u" in con.list_tables()
    finally:
        con.disconnect()


def test_get_schema_rejects_catalog(mem):
    mem.create_table("c", obj=ibis.memtable({"x": [1]}), engine="Memory")
    with pytest.raises(com.UnsupportedOperationError):
        mem.get_schema("c", catalog="somecatalog")


def test_sql_method(mem):
    t = mem.sql("SELECT number AS n FROM numbers(5)")
    assert t.count().execute() == 5
    # ClickHouse/chDB columns are non-nullable by default
    assert t.schema().names == ("n",)
    assert t.schema()["n"].is_integer()


def test_current_database(mem):
    # chDB can't reuse ClickHouse's current_database (it reads clickhouse_connect
    # result_rows); assert the value directly so a regression can't slip through.
    assert mem.current_database == "default"


def test_missing_chdb_raises_actionable_error(monkeypatch):
    # chdb is intentionally absent from the extra; a missing engine must point
    # the user at `pip install chdb`, not raise a bare ModuleNotFoundError.
    monkeypatch.setitem(sys.modules, "chdb", None)
    with pytest.raises(ImportError, match="pip install chdb"):
        _import_chdb()
