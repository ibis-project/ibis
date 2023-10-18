from __future__ import annotations

import duckdb
import pytest
import sqlalchemy as sa

import ibis
from ibis.conftest import LINUX, SANDBOXED


@pytest.fixture(scope="session")
def ext_directory(tmpdir_factory):
    # A session-scoped temp directory to cache extension downloads per session.
    # Coupled with the xdist_group below, this ensures that the extension
    # loading tests always run in the same process and a common temporary
    # directory isolated from other duckdb tests, avoiding issues with
    # downloading extensions in parallel.
    return str(tmpdir_factory.mktemp("exts"))


@pytest.mark.xfail(
    LINUX and SANDBOXED,
    reason="nix on linux cannot download duckdb extensions or data due to sandboxing",
    raises=sa.exc.OperationalError,
)
@pytest.mark.xdist_group(name="duckdb-extensions")
def test_connect_extensions(ext_directory):
    con = ibis.duckdb.connect(
        extensions=["s3", "sqlite"],
        extension_directory=ext_directory,
    )
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
@pytest.mark.xdist_group(name="duckdb-extensions")
def test_load_extension(ext_directory):
    con = ibis.duckdb.connect(extension_directory=ext_directory)
    con.load_extension("s3")
    con.load_extension("sqlite")
    results = con.raw_sql(
        """
        SELECT loaded FROM duckdb_extensions()
        WHERE extension_name = 'httpfs' OR extension_name = 'sqlite'
        """
    ).fetchall()
    assert all(loaded for (loaded,) in results)


def test_cross_db(tmpdir):
    import duckdb

    path1 = str(tmpdir.join("test1.ddb"))
    with duckdb.connect(path1) as con1:
        con1.execute("CREATE SCHEMA foo")
        con1.execute("CREATE TABLE t1 (x BIGINT)")
        con1.execute("CREATE TABLE foo.t1 (x BIGINT)")

    path2 = str(tmpdir.join("test2.ddb"))
    con2 = ibis.duckdb.connect(path2)
    t2 = con2.create_table("t2", schema=ibis.schema(dict(x="int")))

    with con2.begin() as c:
        c.exec_driver_sql(f"ATTACH '{path1}' AS test1 (READ_ONLY)")

    t1_from_con2 = con2.table("t1", schema="test1.main")
    assert t1_from_con2.schema() == t2.schema()
    assert t1_from_con2.execute().equals(t2.execute())

    foo_t1_from_con2 = con2.table("t1", schema="test1.foo")
    assert foo_t1_from_con2.schema() == t2.schema()
    assert foo_t1_from_con2.execute().equals(t2.execute())
