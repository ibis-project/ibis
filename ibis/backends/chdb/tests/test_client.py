"""Connection and catalog behavior specific to the chDB backend."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

import ibis
import ibis.common.exceptions as com
from ibis import util


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


def test_timestamps_ignore_local_timezone():
    # chDB defaults to the host's timezone for the session and for Python UDF
    # values, which CI (UTC) never sees
    code = textwrap.dedent(
        """
        import datetime

        import pandas as pd

        import ibis
        import ibis.expr.datatypes as dt


        @ibis.udf.scalar.python
        def naive(x: int) -> datetime.datetime:
            return datetime.datetime(2026, 7, 22, 1, 2, 3)


        @ibis.udf.scalar.python(
            signature=((dt.int64,), dt.Timestamp(timezone="America/New_York"))
        )
        def naive_ny(x):
            return datetime.datetime(2026, 7, 22, 1, 2, 3)


        @ibis.udf.scalar.python
        def ident(x: datetime.datetime) -> datetime.datetime:
            return x


        con = ibis.chdb.connect()
        ts = ibis.timestamp("2026-07-22 01:02:03")
        t = ibis.memtable({"ts": pd.to_datetime(["2026-07-22 01:02:03"])})
        print(con.execute(ts))
        print(con.execute(t.ts.hour()).tolist())
        print(con.execute(naive(ibis.literal(1))))
        print(con.execute(ident(ts).hour()))
        print(con.execute(naive_ny(ibis.literal(1))))
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "TZ": "America/New_York"},
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.splitlines() == [
        "2026-07-22 01:02:03",
        "[1]",
        "2026-07-22 01:02:03",
        "1",
        "2026-07-22 01:02:03-04:00",
    ]


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


def test_connect_rejects_unknown_kwargs():
    with pytest.raises(TypeError, match="path"):
        ibis.chdb.connect(path="/some/path")


def test_read_csv_keeps_existing_table(mem, tmp_path):
    path = tmp_path / "t.csv"
    path.write_text("x\n1\n")
    mem.create_table("t", obj=ibis.memtable({"x": [1, 2]}), engine="Memory")
    with pytest.raises(RuntimeError, match="TABLE_ALREADY_EXISTS"):
        mem.read_csv(path, table_name="t")
    assert mem.table("t").count().execute() == 2


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


@pytest.mark.parametrize("from_connection", [False, True])
@pytest.mark.parametrize(
    ("version", "found"),
    [("26.11.1rc1", "chdb-core 26.11.1rc1"), (None, "no chdb-core package")],
)
def test_old_chdb_core_raises_actionable_error(
    monkeypatch, chdb_path, version, found, from_connection
):
    import chdb

    monkeypatch.setattr(util, "version", lambda _: version)
    with pytest.raises(ImportError, match=f"chdb>=4\\.5\\.0.*found {found}"):
        if from_connection:
            ibis.chdb.from_connection(chdb.connect(chdb_path))
        else:
            ibis.chdb.connect(chdb_path)


def test_missing_chdb_raises_actionable_error(monkeypatch):
    # chdb is intentionally absent from the extra; a missing engine must point
    # the user at `pip install 'chdb>=4.5.0'`, not raise a bare
    # ModuleNotFoundError.
    monkeypatch.setitem(sys.modules, "chdb", None)
    with pytest.raises(ImportError, match=r"pip install 'chdb>=4\.5\.0'"):
        ibis.chdb.connect()
