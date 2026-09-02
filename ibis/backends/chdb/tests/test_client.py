"""Connection and catalog behavior specific to the chDB backend."""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

import ibis
import ibis.common.exceptions as com


def test_imports_without_clickhouse_connect():
    # the chdb extra doesn't pull clickhouse-connect, so the backend must
    # import cleanly without it (the clickhouse imports are deferred)
    code = textwrap.dedent(
        """
        import sys

        sys.modules["clickhouse_connect"] = None
        import ibis.backends.chdb

        print("ok")
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "ok"


def test_raw_sql(mem):
    res = mem.raw_sql("SELECT 1 AS a, 'x' AS b", fmt="CSV")
    assert str(res).strip() == '1,"x"'


def test_list_tables_like(mem):
    mem.create_table("apple", obj=ibis.memtable({"x": [1]}), engine="Memory")
    mem.create_table("apricot", obj=ibis.memtable({"x": [1]}), engine="Memory")
    mem.create_table("banana", obj=ibis.memtable({"x": [1]}), engine="Memory")
    assert set(mem.list_tables(like="ap")) == {"apple", "apricot"}


def test_multiple_connections_same_path(chdb_path):
    c1 = ibis.chdb.connect(chdb_path)
    c2 = ibis.chdb.connect(chdb_path)
    assert c1.execute(ibis.memtable({"a": [1, 2]}).a.sum()) == 3
    assert c2.name == "chdb"
    c1.disconnect()
    c2.disconnect()


def test_connect_via_url(chdb_path):
    # chdb://<path> must resolve to a path-based connection (not the inherited
    # ClickHouse host/port URL parser).
    con = ibis.connect(f"chdb://{chdb_path}")
    try:
        assert con.name == "chdb"
        assert con.execute(ibis.memtable({"x": [1, 2]}).x.sum()) == 3
    finally:
        con.disconnect()


def test_get_schema_rejects_catalog(mem):
    mem.create_table("c", obj=ibis.memtable({"x": [1]}), engine="Memory")
    with pytest.raises(com.UnsupportedOperationError):
        mem.get_schema("c", catalog="somecatalog")


def test_current_database(chdb_path):
    # chDB can't reuse ClickHouse's current_database (it reads clickhouse_connect
    # result_rows); assert the value directly so a regression can't slip through.
    con = ibis.chdb.connect(chdb_path)
    try:
        assert con.current_database == "default"
    finally:
        con.disconnect()


def test_missing_chdb_raises_actionable_error(monkeypatch):
    # chdb is intentionally absent from the extra; a missing engine must point
    # the user at `pip install chdb`, not raise a bare ModuleNotFoundError.
    monkeypatch.setitem(sys.modules, "chdb", None)
    with pytest.raises(ImportError, match="pip install chdb"):
        ibis.chdb.connect()
