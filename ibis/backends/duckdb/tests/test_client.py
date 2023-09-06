from __future__ import annotations

import duckdb
import pytest
import sqlalchemy as sa

import ibis
from ibis.conftest import LINUX, SANDBOXED


@pytest.mark.xfail(
    LINUX and SANDBOXED,
    reason="nix on linux cannot download duckdb extensions or data due to sandboxing",
    raises=sa.exc.OperationalError,
)
def test_connect_extensions():
    con = ibis.duckdb.connect(extensions=["s3", "sqlite"])
    results = con.raw_sql(
        """
        SELECT loaded FROM duckdb_extensions()
        WHERE extension_name = 'httpfs' OR extension_name = 'sqlite'
        """
    ).fetchall()
    assert all(loaded for (loaded,) in results)


@pytest.mark.xfail(
    LINUX and SANDBOXED,
    reason="nix on linux cannot download duckdb extensions or data due to sandboxing",
    raises=duckdb.IOException,
)
def test_load_extension():
    con = ibis.duckdb.connect()
    con.load_extension("s3")
    con.load_extension("sqlite")
    results = con.raw_sql(
        """
        SELECT loaded FROM duckdb_extensions()
        WHERE extension_name = 'httpfs' OR extension_name = 'sqlite'
        """
    ).fetchall()
    assert all(loaded for (loaded,) in results)
