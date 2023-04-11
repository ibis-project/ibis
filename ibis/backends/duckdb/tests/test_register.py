import os
import sqlite3
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pandas.testing as tm
import pyarrow as pa
import pytest

import ibis
import ibis.common.exceptions as exc


def test_read_csv(data_directory):
    t = ibis.read_csv(data_directory / "functional_alltypes.csv")
    assert t.count().execute()


def test_read_parquet(data_directory):
    t = ibis.read_parquet(data_directory / "functional_alltypes.parquet")
    assert t.count().execute()


@pytest.mark.xfail_version(
    duckdb=["duckdb<0.7.0"], reason="read_json_auto doesn't exist", raises=exc.IbisError
)
def test_read_json(data_directory, tmp_path):
    pqt = ibis.read_parquet(data_directory / "functional_alltypes.parquet")

    path = tmp_path.joinpath("ft.json")
    path.write_text(pqt.execute().to_json(orient="records", lines=True))

    jst = ibis.read_json(path)

    nrows = pqt.count().execute()
    assert nrows
    assert nrows == jst.count().execute()


def test_temp_directory(tmp_path):
    query = "SELECT current_setting('temp_directory')"

    # 1. in-memory + no temp_directory specified
    con = ibis.duckdb.connect()
    with con.begin() as c:
        value = c.exec_driver_sql(query).scalar()
        assert value  # we don't care what the specific value is

    temp_directory = Path(tempfile.gettempdir()) / "duckdb"

    # 2. in-memory + temp_directory specified
    con = ibis.duckdb.connect(temp_directory=temp_directory)
    with con.begin() as c:
        value = c.exec_driver_sql(query).scalar()
    assert value == str(temp_directory)

    # 3. on-disk + no temp_directory specified
    # untested, duckdb sets the temp_directory to something implementation
    # defined

    # 4. on-disk + temp_directory specified
    con = ibis.duckdb.connect(tmp_path / "test2.ddb", temp_directory=temp_directory)
    with con.begin() as c:
        value = c.exec_driver_sql(query).scalar()
    assert value == str(temp_directory)


@pytest.fixture(scope="session")
def pgurl():  # pragma: no cover
    pgcon = ibis.postgres.connect(user="postgres", password="postgres")
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 1.0], "y": ["a", "b", "c", "a"]})
    s = ibis.schema(dict(x="float64", y="str"))

    pgcon.create_table("duckdb_test", df, s, force=True)
    yield pgcon.con.url
    pgcon.drop_table("duckdb_test", force=True)


@pytest.mark.skipif(
    os.environ.get("DUCKDB_POSTGRES") is None, reason="avoiding CI shenanigans"
)
def test_read_postgres(con, pgurl):  # pragma: no cover
    table = con.read_postgres(
        f"postgres://{pgurl.username}:{pgurl.password}@{pgurl.host}:{pgurl.port}",
        table_name="duckdb_test",
    )
    assert table.count().execute()


def test_read_sqlite(con, tmp_path):
    path = tmp_path / "test.db"

    sqlite_con = sqlite3.connect(str(path))
    sqlite_con.execute("CREATE TABLE t AS SELECT 1 a UNION SELECT 2 UNION SELECT 3")

    ft = con.read_sqlite(path, table_name="t")
    assert ft.count().execute()

    with pytest.raises(ValueError):
        con.read_sqlite(path)


def test_read_sqlite_no_table_name(con, tmp_path):
    path = tmp_path / "test.db"

    sqlite3.connect(str(path))

    assert path.exists()

    with pytest.raises(ValueError):
        con.read_sqlite(path)


def test_register_sqlite(con, tmp_path):
    path = tmp_path / "test.db"

    sqlite_con = sqlite3.connect(str(path))
    sqlite_con.execute("CREATE TABLE t AS SELECT 1 a UNION SELECT 2 UNION SELECT 3")
    ft = con.register(f"sqlite://{path}", "t")
    assert ft.count().execute()


def test_read_in_memory(con):
    df_arrow = pa.table({"a": ["a"], "b": [1]})
    df_pandas = pd.DataFrame({"a": ["a"], "b": [1]})
    con.read_in_memory(df_arrow, table_name="df_arrow")
    con.read_in_memory(df_pandas, table_name="df_pandas")

    assert "df_arrow" in con.list_tables()
    assert "df_pandas" in con.list_tables()


def test_re_read_in_memory_overwrite(con):
    df_pandas_1 = pd.DataFrame({"a": ["a"], "b": [1], "d": ["hi"]})
    df_pandas_2 = pd.DataFrame({"a": [1], "c": [1.4]})

    table = con.read_in_memory(df_pandas_1, table_name="df")
    assert len(table.columns) == 3
    assert table.schema() == ibis.schema([("a", "str"), ("b", "int"), ("d", "str")])

    table = con.read_in_memory(df_pandas_2, table_name="df")
    assert len(table.columns) == 2
    assert table.schema() == ibis.schema([("a", "int"), ("c", "float")])


def test_memtable_with_nullable_dtypes():
    data = pd.DataFrame(
        {
            "a": pd.Series(["a", None, "c"], dtype="string"),
            "b": pd.Series([None, 1, 2], dtype="Int8"),
            "c": pd.Series([0, None, 2], dtype="Int16"),
            "d": pd.Series([0, 1, None], dtype="Int32"),
            "e": pd.Series([None, None, -1], dtype="Int64"),
            "f": pd.Series([None, 1, 2], dtype="UInt8"),
            "g": pd.Series([0, None, 2], dtype="UInt16"),
            "h": pd.Series([0, 1, None], dtype="UInt32"),
            "i": pd.Series([None, None, 42], dtype="UInt64"),
            "j": pd.Series([None, False, True], dtype="boolean"),
        }
    )
    expr = ibis.memtable(data)
    res = expr.execute()
    assert len(res) == len(data)


def test_memtable_with_nullable_pyarrow_string():
    pytest.importorskip("pyarrow")
    data = pd.DataFrame({"a": pd.Series(["a", None, "c"], dtype="string[pyarrow]")})
    expr = ibis.memtable(data)
    res = expr.execute()
    assert len(res) == len(data)


def test_memtable_with_nullable_pyarrow_not_string():
    pytest.importorskip("pyarrow")

    data = pd.DataFrame(
        {
            "b": pd.Series([None, 1, 2], dtype="int8[pyarrow]"),
            "c": pd.Series([0, None, 2], dtype="int16[pyarrow]"),
            "d": pd.Series([0, 1, None], dtype="int32[pyarrow]"),
            "e": pd.Series([None, None, -1], dtype="int64[pyarrow]"),
            "f": pd.Series([None, 1, 2], dtype="uint8[pyarrow]"),
            "g": pd.Series([0, None, 2], dtype="uint16[pyarrow]"),
            "h": pd.Series([0, 1, None], dtype="uint32[pyarrow]"),
            "i": pd.Series([None, None, 42], dtype="uint64[pyarrow]"),
            "j": pd.Series([None, False, True], dtype="boolean[pyarrow]"),
        }
    )
    expr = ibis.memtable(data)
    res = expr.execute()
    assert len(res) == len(data)


def test_set_temp_dir(tmp_path):
    path = tmp_path / "foo" / "bar"
    ibis.duckdb.connect(temp_directory=path)
    assert path.exists()


def test_s3_403_fallback(con, httpserver, monkeypatch):
    # monkeypatch to avoid downloading extensions in tests
    monkeypatch.setattr(con, "_load_extensions", lambda x: True)

    # Throw a 403 to trigger fallback to pyarrow.dataset
    httpserver.expect_request("/myfile").respond_with_data(
        "Forbidden", status=403, content_type="text/plain"
    )

    # Since the URI is nonsense to pyarrow, expect an error, but raises from
    # pyarrow, which indicates the fallback worked
    with pytest.raises(pa.lib.ArrowInvalid):
        con.read_parquet(httpserver.url_for("/myfile"))


@pytest.mark.xfail_version(
    duckdb=["duckdb<=0.7.1"],
    reason="""
the fix for this (issue #5879) caused a serious performance regression in the repr.
added this xfail in #5959, which also reverted the bugfix that caused the regression.

the issue was fixed upstream in duckdb in https://github.com/duckdb/duckdb/pull/6978
    """,
)
def test_register_numpy_str(con):
    data = pd.DataFrame({"a": [np.str_("xyz"), None]})
    result = con.read_in_memory(data)
    tm.assert_frame_equal(result.execute(), data)
