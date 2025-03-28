from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pytest
from pytest import param

import ibis
import ibis.expr.operations as ops
from ibis.conftest import not_windows


def test_attach_file(tmp_path):
    dbpath = str(tmp_path / "attached.db")
    path_client = ibis.sqlite.connect(dbpath)
    client = ibis.sqlite.connect()
    try:
        path_client.create_table("test", schema=ibis.schema(dict(a="int")))

        assert not client.list_tables()

        client.attach("baz", Path(dbpath))
        client.attach("bar", dbpath)

        foo_tables = client.list_tables(database="baz")
        bar_tables = client.list_tables(database="bar")

        assert foo_tables == ["test"]
        assert foo_tables == bar_tables
    finally:
        client.disconnect()
        path_client.disconnect()


def test_builtin_scalar_udf(con):
    @ibis.udf.scalar.builtin
    def zeroblob(n: int) -> bytes:
        """Return a length `n` blob of zero bytes."""

    n = 42
    expr = zeroblob(n)
    result = con.execute(expr)
    assert result == b"\x00" * n


def test_builtin_agg_udf(con):
    @ibis.udf.agg.builtin
    def total(x) -> float:
        """Totally total."""

    expr = total(con.tables.functional_alltypes.limit(2).select(n=ibis.null()).n)
    result = con.execute(expr)
    assert result == 0.0


@pytest.mark.parametrize(
    "url, ext",
    [
        param(lambda p: p, "sqlite", id="no-scheme-sqlite-ext"),
        param(lambda p: p, "db", id="no-scheme-db-ext"),
        param(lambda p: f"sqlite://{p}", "db", id="absolute-path"),
        param(
            lambda p: f"sqlite://{os.path.relpath(p)}",
            "db",
            # hard to test in CI since tmpdir & cwd are on different drives
            marks=[not_windows],
            id="relative-path",
        ),
        param(lambda _: "sqlite://", "db", id="in-memory-empty"),
        param(lambda _: "sqlite://:memory:", "db", id="in-memory-explicit"),
    ],
)
def test_connect(url, ext, tmp_path):
    path = os.path.abspath(tmp_path / f"test.{ext}")

    sqlite3.connect(path).close()

    con = ibis.connect(url(path))
    try:
        assert con.execute(ibis.literal(1)) == 1
    finally:
        con.disconnect()


def test_has_operation(con):
    # Core operations handled in non-standard ways
    for op in [ops.Project, ops.Filter, ops.Sort, ops.Aggregate]:
        assert con.has_operation(op)
    # Handled by base class rewrite
    assert con.has_operation(ops.Capitalize)
    # Handled by compiler-specific rewrite
    assert con.has_operation(ops.Sample)
    # Handled by visit_* method
    assert con.has_operation(ops.Cast)


def test_list_temp_tables_by_default(con):
    name = ibis.util.gen_name("sqlite_temp_table")
    con.create_table(name, schema={"a": "int"}, temp=True)
    assert name in con.list_tables(database="temp")
    assert name in con.list_tables()
