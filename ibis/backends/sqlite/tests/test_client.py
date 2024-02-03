from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pytest
from pytest import param

import ibis
from ibis.conftest import not_windows


def test_attach_file(tmp_path):
    dbpath = str(tmp_path / "attached.db")
    path_client = ibis.sqlite.connect(dbpath)
    path_client.create_table("test", schema=ibis.schema(dict(a="int")))

    client = ibis.sqlite.connect()

    assert not client.list_tables()

    client.attach("baz", Path(dbpath))
    client.attach("bar", dbpath)

    foo_tables = client.list_tables(database="baz")
    bar_tables = client.list_tables(database="bar")

    assert foo_tables == ["test"]
    assert foo_tables == bar_tables


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

    expr = total(con.tables.functional_alltypes.limit(2).select(n=ibis.NA).n)
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
    with sqlite3.connect(path):
        pass
    con = ibis.connect(url(path))
    one = ibis.literal(1)
    assert con.execute(one) == 1
