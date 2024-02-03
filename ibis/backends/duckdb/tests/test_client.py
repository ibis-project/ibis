from __future__ import annotations

import os
import subprocess
import sys

import duckdb
import pandas as pd
import pyarrow as pa
import pytest
from pytest import param

import ibis
import ibis.expr.datatypes as dt
from ibis.conftest import LINUX, SANDBOXED, not_windows
from ibis.util import gen_name


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
    raises=duckdb.IOException,
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
    path1 = str(tmpdir.join("test1.ddb"))
    with duckdb.connect(path1) as con1:
        con1.execute("CREATE SCHEMA foo")
        con1.execute("CREATE TABLE t1 (x BIGINT)")
        con1.execute("CREATE TABLE foo.t1 (x BIGINT)")

    path2 = str(tmpdir.join("test2.ddb"))
    con2 = ibis.duckdb.connect(path2)
    t2 = con2.create_table("t2", schema=ibis.schema(dict(x="int")))

    con2.attach(path1, name="test1", read_only=True)

    t1_from_con2 = con2.table("t1", schema="main", database="test1")
    assert t1_from_con2.schema() == t2.schema()
    assert t1_from_con2.execute().equals(t2.execute())

    foo_t1_from_con2 = con2.table("t1", schema="foo", database="test1")
    assert foo_t1_from_con2.schema() == t2.schema()
    assert foo_t1_from_con2.execute().equals(t2.execute())


def test_attach_detach(tmpdir):
    path1 = str(tmpdir.join("test1.ddb"))
    with duckdb.connect(path1):
        pass

    path2 = str(tmpdir.join("test2.ddb"))
    con2 = ibis.duckdb.connect(path2)

    # default name
    name = "test1"
    assert name not in con2.list_databases()

    con2.attach(path1)
    assert name in con2.list_databases()

    con2.detach(name)
    assert name not in con2.list_databases()

    # passed-in name
    name = "test_foo"
    assert name not in con2.list_databases()

    con2.attach(path1, name=name)
    assert name in con2.list_databases()

    con2.detach(name)
    assert name not in con2.list_databases()

    with pytest.raises(duckdb.BinderException):
        con2.detach(name)


@pytest.mark.parametrize(
    ("scale", "expected_scale"),
    [
        param(None, 6, id="default"),
        param(0, 0, id="seconds"),
        param(3, 3, id="millis"),
        param(6, 6, id="micros"),
        param(9, 9, id="nanos"),
    ],
)
def test_create_table_with_timestamp_scales(con, scale, expected_scale):
    schema = ibis.schema(dict(ts=dt.Timestamp(scale=scale)))
    expected = ibis.schema(dict(ts=dt.Timestamp(scale=expected_scale)))
    name = gen_name("duckdb_timestamp_scale")
    t = con.create_table(name, schema=schema, temp=True)
    assert t.schema() == expected


def test_config_options(con):
    a_first = {"a": [None, 1]}
    a_last = {"a": [1, None]}
    nulls_first = pa.Table.from_pydict(a_first, schema=pa.schema([("a", pa.float64())]))
    nulls_last = pa.Table.from_pydict(a_last, schema=pa.schema([("a", pa.float64())]))

    t = ibis.memtable(a_last)

    expr = t.order_by("a")

    assert con.to_pyarrow(expr) == nulls_last

    con.settings["null_order"] = "nulls_first"

    assert con.to_pyarrow(expr) == nulls_first


def test_config_options_bad_option(con):
    with pytest.raises(duckdb.CatalogException):
        con.settings["not_a_valid_option"] = "oopsie"

    with pytest.raises(KeyError):
        con.settings["i_didnt_set_this"]


def test_insert(con):
    name = ibis.util.guid()

    t = con.create_table(name, schema=ibis.schema({"a": "int64"}), temp=True)

    con.insert(name, obj=pd.DataFrame({"a": [1, 2]}))
    assert t.count().execute() == 2

    con.insert(name, obj=pd.DataFrame({"a": [1, 2]}))
    assert t.count().execute() == 4

    con.insert(name, obj=pd.DataFrame({"a": [1, 2]}), overwrite=True)
    assert t.count().execute() == 2

    con.insert(name, t)
    assert t.count().execute() == 4

    con.insert(name, [{"a": 1}, {"a": 2}], overwrite=True)
    assert t.count().execute() == 2

    con.insert(name, [(1,), (2,)])
    assert t.count().execute() == 4

    con.insert(name, {"a": [1, 2]}, overwrite=True)
    assert t.count().execute() == 2


def test_to_other_sql(con, snapshot):
    pytest.importorskip("snowflake.connector")

    t = con.table("functional_alltypes")

    sql = ibis.to_sql(t, dialect="snowflake")
    snapshot.assert_match(sql, "out.sql")


def test_insert_preserves_column_case(con):
    name1 = ibis.util.guid()
    name2 = ibis.util.guid()

    df1 = pd.DataFrame([[1], [2], [3], [4]], columns=["FTHG"])
    df2 = pd.DataFrame([[5], [6], [7], [8]], columns=["FTHG"])

    t1 = con.create_table(name1, df1, temp=True)
    assert t1.count().execute() == 4

    t2 = con.create_table(name2, df2, temp=True)
    con.insert(name1, t2)
    assert t1.count().execute() == 8


def test_default_backend():
    # use subprocess to avoid mutating state across tests
    script = """\
import pandas as pd

import ibis

df = pd.DataFrame({"a": [1, 2, 3]})

t = ibis.memtable(df)

expr = t.a.sum()

# run twice to ensure that we hit the optimizations in
# `_default_backend`
for _ in range(2):
    assert expr.execute() == df.a.sum()"""

    assert ibis.options.default_backend is None
    subprocess.run([sys.executable, "-c", script], check=True)
    assert ibis.options.default_backend is None


@pytest.mark.parametrize(
    "url",
    [
        param(lambda p: p, id="no-scheme-duckdb-ext"),
        param(lambda p: f"duckdb://{p}", id="absolute-path"),
        param(
            lambda p: f"duckdb://{os.path.relpath(p)}",
            # hard to test in CI since tmpdir & cwd are on different drives
            marks=[not_windows],
            id="relative-path",
        ),
        param(lambda _: "duckdb://", id="in-memory-empty"),
        param(lambda _: "duckdb://:memory:", id="in-memory-explicit"),
        param(lambda p: f"duckdb://{p}?read_only=1", id="duckdb_read_write_int"),
        param(lambda p: f"duckdb://{p}?read_only=False", id="duckdb_read_write_upper"),
        param(lambda p: f"duckdb://{p}?read_only=false", id="duckdb_read_write_lower"),
    ],
)
def test_connect_duckdb(url, tmp_path):
    path = os.path.abspath(tmp_path / "test.duckdb")
    with duckdb.connect(path):
        pass
    con = ibis.connect(url(path))
    one = ibis.literal(1)
    assert con.execute(one) == 1


@pytest.mark.parametrize(
    "out_method, extension", [("to_csv", "csv"), ("to_parquet", "parquet")]
)
def test_connect_local_file(out_method, extension, test_employee_data_1, tmp_path):
    getattr(test_employee_data_1, out_method)(tmp_path / f"out.{extension}")
    con = ibis.connect(tmp_path / f"out.{extension}")
    t = next(iter(con.tables.values()))
    assert not t.head().execute().empty


@not_windows
def test_invalid_connect(tmp_path):
    url = f"duckdb://{tmp_path}?read_only=invalid_value"
    with pytest.raises(ValueError):
        ibis.connect(url)
