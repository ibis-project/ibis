from __future__ import annotations

import os
import sqlite3

import duckdb
import numpy as np
import pandas as pd
import pandas.testing as tm
import pyarrow as pa
import pytest
from packaging.version import parse as parse_version
from pytest import param

import ibis
import ibis.expr.datatypes as dt
from ibis.conftest import ARM64, LINUX, MACOS, SANDBOXED


def test_read_csv(con, data_dir):
    t = con.read_csv(data_dir / "csv" / "functional_alltypes.csv")
    assert t.count().execute()


def test_read_csv_with_columns(con, data_dir):
    t = con.read_csv(
        data_dir / "csv" / "awards_players.csv",
        header=True,
        columns={
            "playerID": "VARCHAR",
            "awardID": "VARCHAR",
            "yearID": "DATE",
            "lgID": "VARCHAR",
            "tie": "VARCHAR",
            "notes": "VARCHAR",
        },
        dateformat="%Y",
    )
    assert t.count().execute()


def test_read_parquet(con, data_dir):
    t = con.read_parquet(data_dir / "parquet" / "functional_alltypes.parquet")
    assert t.count().execute()


@pytest.mark.xfail(
    LINUX and SANDBOXED,
    reason="nix on linux cannot download duckdb extensions or data due to sandboxing",
)
def test_load_spatial_when_geo_column(tmpdir):
    pytest.importorskip("geopandas")
    pytest.importorskip("shapely")

    path = str(tmpdir.join("test_load_spatial.ddb"))

    with duckdb.connect(
        # windows is horrible and cannot download in parallel without
        # clobbering existing files, so give a temporary custom directory for
        # extensions
        path,
        config={"extension_directory": str(tmpdir.join("extensions"))},
    ) as con:
        con.install_extension("spatial")
        con.load_extension("spatial")
        con.execute(
            # create a table with a geom column
            """
            CREATE or REPLACE TABLE samples (name VARCHAR, geom GEOMETRY);

            INSERT INTO samples VALUES
              ('Point', ST_GeomFromText('POINT(-100 40)')),
              ('Linestring', ST_GeomFromText('LINESTRING(0 0, 1 1, 2 1, 2 2)')),
              ('Polygon', ST_GeomFromText('POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))'));
            """
        )

    # load data from ibis and check for spatial extension
    con = ibis.duckdb.connect(path)

    query = """\
SELECT extension_name AS name
FROM duckdb_extensions()
WHERE installed AND loaded"""

    assert "spatial" not in con.sql(query).name.to_pandas().values

    # trigger spatial extension load
    assert not con.tables.samples.head(1).geom.to_pandas().empty

    assert "spatial" in con.sql(query).name.to_pandas().values


@pytest.mark.usefixtures("gpd")
def test_read_geo_to_pyarrow(con, data_dir):
    shapely = pytest.importorskip("shapely")

    t = con.read_geo(data_dir / "geojson" / "zones.geojson")
    raw_geometry = t.head().to_pyarrow()["geom"].to_pandas()
    assert len(shapely.from_wkb(raw_geometry))


def test_read_geo_to_geopandas(con, data_dir, gpd):
    t = con.read_geo(data_dir / "geojson" / "zones.geojson")
    gdf = t.head().to_pandas()
    assert isinstance(gdf, gpd.GeoDataFrame)


def test_read_geo_from_url(monkeypatch):
    con = ibis.duckdb.connect()

    loaded_exts = []
    monkeypatch.setattr(con, "_load_extensions", lambda x, **_: loaded_exts.extend(x))

    with pytest.raises((duckdb.IOException, duckdb.CatalogException)):
        # The read will fail, either because the URL is bogus (which it is) or
        # because the current connection doesn't have the spatial extension
        # installed and so the call to `st_read` will raise a catalog error.
        con.read_geo("https://...")

    assert "spatial" in loaded_exts
    assert "httpfs" in loaded_exts


def test_read_json(con, data_dir, tmp_path):
    pqt = con.read_parquet(data_dir / "parquet" / "functional_alltypes.parquet")

    path = tmp_path.joinpath("ft.json")
    path.write_text(pqt.execute().to_json(orient="records", lines=True))

    jst = con.read_json(path)

    nrows = pqt.count().execute()
    assert nrows
    assert nrows == jst.count().execute()


@pytest.fixture(scope="session")
def pgurl():  # pragma: no cover
    pgcon = ibis.postgres.connect(
        user="postgres",
        password="postgres",  # noqa: S106
        host="localhost",
    )

    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 1.0], "y": ["a", "b", "c", "a"]})

    pgcon.create_table("duckdb_test", df, overwrite=True)
    yield pgcon.con.info

    pgcon.drop_table("duckdb_test", force=True)


@pytest.mark.skipif(
    os.environ.get("DUCKDB_POSTGRES") is None, reason="avoiding CI shenanigans"
)
def test_read_postgres(con, pgurl):  # pragma: no cover
    # we don't run this test in CI, only locally, to avoid bringing a postgres
    # container up just for this test.  To run locally set env variable to True
    # and once a postgres container is up run the test.
    table = con.read_postgres(
        f"postgres://{pgurl.user}:{pgurl.password}@{pgurl.host}:{pgurl.port}",
        table_name="duckdb_test",
    )
    assert table.count().execute()


@pytest.fixture(scope="session")
def mysqlurl():  # pragma: no cover
    mysqlcon = ibis.mysql.connect(
        user="ibis",
        password="ibis",  # noqa: S106
        database="ibis_testing",
    )

    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 1.0], "y": ["a", "b", "c", "a"]})
    s = ibis.schema(dict(x="float64", y="str"))

    mysqlcon.create_table("duckdb_test", df, schema=s, overwrite=True)
    yield mysqlcon.con
    mysqlcon.drop_table("duckdb_test", force=True)


@pytest.mark.skipif(
    os.environ.get("DUCKDB_MYSQL") is None, reason="avoiding CI shenanigans"
)
def test_read_mysql(con, mysqlurl):  # pragma: no cover
    # to run this test run first the mysql test suit to get the ibis-testing
    # we don't run this test in CI, only locally, to avoid bringing a mysql
    # container up just for this test.  To run locally set env variable to True
    # and once a mysql container is up run the test.

    # TODO(ncclementi) replace for mysqlurl.host when this is fix
    # https://github.com/duckdb/duckdb_mysql/issues/44
    hostname = "127.0.0.1"

    table = con.read_mysql(
        f"mysql://{mysqlurl.user.decode()}:{mysqlurl.password.decode()}@{hostname}:{mysqlurl.port}/ibis_testing",
        catalog="mysqldb",
        table_name="duckdb_test",
    )
    assert table.count().execute()


@pytest.mark.xfail(
    LINUX and SANDBOXED,
    reason="nix on linux cannot download duckdb extensions or data due to sandboxing",
    raises=duckdb.IOException,
)
def test_read_sqlite(con, tmp_path):
    path = tmp_path / "test.db"

    scon = sqlite3.connect(str(path))
    try:
        with scon:
            scon.execute("CREATE TABLE t AS SELECT 1 a UNION SELECT 2 UNION SELECT 3")
    finally:
        scon.close()

    ft = con.read_sqlite(path, table_name="t")
    assert ft.count().execute()


def test_read_sqlite_no_table_name(con, tmp_path):
    path = tmp_path / "test.db"

    scon = sqlite3.connect(str(path))
    try:
        assert path.exists()

        with pytest.raises(ValueError):
            con.read_sqlite(path)
    finally:
        scon.close()


# Because we create a new connection and the test requires loading/installing a
# DuckDB extension, we need to xfail these on Nix.
@pytest.mark.xfail(
    LINUX and SANDBOXED,
    reason="nix on linux cannot download duckdb extensions or data due to sandboxing",
    raises=duckdb.IOException,
)
def test_attach_sqlite(data_dir, tmp_path):
    import sqlite3

    # Create a new connection here because we already have the `ibis_testing`
    # tables loaded in to the `con` fixture.
    con = ibis.duckdb.connect()

    test_db_path = tmp_path / "test.db"
    scon = sqlite3.connect(test_db_path)
    try:
        with scon:
            scon.executescript((data_dir.parent / "schema" / "sqlite.sql").read_text())

        con.attach_sqlite(test_db_path)
        assert set(con.list_tables()) >= {
            "functional_alltypes",
            "awards_players",
            "batting",
            "diamonds",
        }

        fa = con.tables.functional_alltypes
        assert len(set(fa.schema().types)) > 1

        # overwrite existing sqlite_db and force schema to all strings
        con.attach_sqlite(test_db_path, overwrite=True, all_varchar=True)
        assert set(con.list_tables()) >= {
            "functional_alltypes",
            "awards_players",
            "batting",
            "diamonds",
        }

        fa = con.tables.functional_alltypes
        types = fa.schema().types
        assert len(set(types)) == 1
        assert dt.String(nullable=True) in set(types)
    finally:
        scon.close()


def test_memtable_with_nullable_dtypes(con):
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
    res = con.execute(expr)
    assert len(res) == len(data)


def test_memtable_with_nullable_pyarrow_string(con):
    pytest.importorskip("pyarrow")
    data = pd.DataFrame({"a": pd.Series(["a", None, "c"], dtype="string[pyarrow]")})
    expr = ibis.memtable(data)
    res = con.execute(expr)
    assert len(res) == len(data)


def test_memtable_with_nullable_pyarrow_not_string(con):
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
    res = con.execute(expr)
    assert len(res) == len(data)


@pytest.mark.parametrize(
    "database",
    [lambda parent: parent / "test.ddb", lambda _: ":memory:"],
    ids=["disk", "memory"],
)
def test_temp_dir_set(tmp_path, database):
    temp_directory = tmp_path / "does" / "not" / "exist"
    temp_directory.mkdir(parents=True, exist_ok=True)
    con = ibis.duckdb.connect(database(tmp_path), temp_directory=temp_directory)
    assert con.settings["temp_directory"] == str(temp_directory)


@pytest.mark.xfail(
    SANDBOXED and LINUX,
    reason=(
        "nix on linux cannot download duckdb extensions or data due to sandboxing; "
        "duckdb will try to automatically install and load read_parquet"
    ),
    raises=(duckdb.Error, duckdb.IOException),
)
@pytest.mark.skipif(
    SANDBOXED and MACOS and ARM64, reason="raises a RuntimeError on nix macos arm64"
)
def test_s3_403_fallback(con, httpserver, monkeypatch):
    # monkeypatch to avoid downloading extensions in tests
    monkeypatch.setattr(con, "_load_extensions", lambda _: True)

    # Throw a 403 to trigger fallback to pyarrow.dataset
    path = "/invalid.parquet"
    httpserver.expect_request(path).respond_with_data(
        status=403, content_type="application/vnd.apache.parquet"
    )

    # Since the URI is nonsense to pyarrow, expect an error, but raises from
    # pyarrow, which indicates the fallback worked
    url = httpserver.url_for(path)
    with pytest.raises(pa.lib.ArrowInvalid):
        con.read_parquet(url)


def test_register_numpy_str(con):
    data = pd.DataFrame({"a": [np.str_("xyz"), None]})
    result = ibis.memtable(data)
    tm.assert_frame_equal(con.execute(result), data)


def test_memtable_recordbatchreader_raises(con):
    table = pa.Table.from_batches(
        map(pa.RecordBatch.from_pydict, [{"x": [1, 2]}, {"x": [3, 4]}])
    )
    reader = table.to_reader()

    with pytest.raises(TypeError):
        ibis.memtable(reader)

    t = ibis.memtable(reader.read_all())

    # First execute is fine
    res = con.execute(t)
    tm.assert_frame_equal(res, table.to_pandas())


def test_csv_with_slash_n_null(con, tmp_path):
    data_path = tmp_path / "data.csv"
    data_path.write_text("a\n1\n3\n\\N\n")
    t = con.read_csv(data_path, nullstr="\\N")
    col = t.a.execute()
    assert pd.isna(col.iat[-1])


@pytest.mark.xfail(
    LINUX and SANDBOXED, reason="nix can't hit GCS because it is sandboxed."
)
@pytest.mark.parametrize(
    "extensions",
    [
        param([], id="none"),
        param(
            ["httpfs"],
            marks=[
                pytest.mark.xfail(
                    parse_version("0.10.0")
                    <= parse_version(duckdb.__version__)
                    < parse_version("0.10.2"),
                    reason="https://github.com/duckdb/duckdb/issues/10698",
                    raises=duckdb.HTTPException,
                )
            ],
            id="httpfs",
        ),
    ],
)
def test_register_filesystem_gcs(extensions):
    fsspec = pytest.importorskip("fsspec")
    pytest.importorskip("gcsfs")

    con = ibis.duckdb.connect()

    for ext in extensions:
        con.load_extension(ext)

    gcs = fsspec.filesystem("gcs")

    con.register_filesystem(gcs)
    band_members = con.read_csv(
        "gcs://ibis-examples/data/band_members.csv.gz", table_name="band_members"
    )

    assert band_members.count().to_pyarrow()


def test_memtable_null_column_parquet_dtype_roundtrip(con, tmp_path):
    before = ibis.memtable({"a": [None, None, None]}, schema={"a": "string"})
    con.to_parquet(before, tmp_path / "tmp.parquet")
    after = con.read_parquet(tmp_path / "tmp.parquet")

    assert before.a.type() == after.a.type()


def test_read_json_no_auto_detection(con, tmp_path):
    ndjson_data = """
    {"year": 2007}
    {"year": 2008}
    {"year": 2009}
    """
    path = tmp_path.joinpath("test.ndjson")
    path.write_text(ndjson_data)

    t = con.read_json(path, auto_detect=False, columns={"year": "varchar"})
    assert t.year.type() == dt.string
