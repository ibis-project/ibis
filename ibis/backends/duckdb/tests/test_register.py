from __future__ import annotations

import os
import sqlite3
import tempfile
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pandas.testing as tm
import pyarrow as pa
import pytest

import ibis
import ibis.expr.datatypes as dt
from ibis.conftest import LINUX, SANDBOXED


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


def test_read_geo_from_url(con, monkeypatch):
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


def test_temp_directory(tmp_path):
    query = "SELECT current_setting('temp_directory')"

    # 1. in-memory + no temp_directory specified
    con = ibis.duckdb.connect()

    value = con.raw_sql(query).fetchone()[0]
    assert value  # we don't care what the specific value is

    temp_directory = Path(tempfile.gettempdir()) / "duckdb"

    # 2. in-memory + temp_directory specified
    con = ibis.duckdb.connect(temp_directory=temp_directory)
    value = con.raw_sql(query).fetchone()[0]
    assert value == str(temp_directory)

    # 3. on-disk + no temp_directory specified
    # untested, duckdb sets the temp_directory to something implementation
    # defined

    # 4. on-disk + temp_directory specified
    con = ibis.duckdb.connect(tmp_path / "test2.ddb", temp_directory=temp_directory)
    value = con.raw_sql(query).fetchone()[0]
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


@pytest.mark.xfail(
    LINUX and SANDBOXED,
    reason="nix on linux cannot download duckdb extensions or data due to sandboxing",
    raises=duckdb.IOException,
)
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


@pytest.mark.xfail(
    LINUX and SANDBOXED,
    reason="nix on linux cannot download duckdb extensions or data due to sandboxing",
    raises=duckdb.IOException,
)
def test_register_sqlite(con, tmp_path):
    path = tmp_path / "test.db"

    sqlite_con = sqlite3.connect(str(path))
    sqlite_con.execute("CREATE TABLE t AS SELECT 1 a UNION SELECT 2 UNION SELECT 3")
    ft = con.register(f"sqlite://{path}", "t")
    assert ft.count().execute()


# Because we create a new connection and the test requires loading/installing a
# DuckDB extension, we need to xfail these on Nix.
@pytest.mark.xfail(
    LINUX and SANDBOXED,
    reason="nix on linux cannot download duckdb extensions or data due to sandboxing",
    raises=duckdb.IOException,
)
def test_attach_sqlite(data_dir, tmp_path):
    import sqlite3

    test_db_path = tmp_path / "test.db"
    with sqlite3.connect(test_db_path) as scon:
        for line in (
            Path(data_dir.parent / "schema" / "sqlite.sql").read_text().split(";")
        ):
            scon.execute(line)

    # Create a new connection here because we already have the `ibis_testing`
    # tables loaded in to the `con` fixture.
    con = ibis.duckdb.connect()

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


def test_set_temp_dir(tmp_path):
    path = tmp_path / "foo" / "bar"
    ibis.duckdb.connect(temp_directory=path)
    assert path.exists()


@pytest.mark.xfail(
    LINUX and SANDBOXED,
    reason=(
        "nix on linux cannot download duckdb extensions or data due to sandboxing; "
        "duckdb will try to automatically install and load read_parquet"
    ),
    raises=(duckdb.Error, duckdb.IOException),
)
def test_s3_403_fallback(con, httpserver, monkeypatch):
    # monkeypatch to avoid downloading extensions in tests
    monkeypatch.setattr(con, "_load_extensions", lambda _: True)

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


def test_register_recordbatchreader_warns(con):
    table = pa.Table.from_batches(
        [
            pa.RecordBatch.from_pydict({"x": [1, 2]}),
            pa.RecordBatch.from_pydict({"x": [3, 4]}),
        ]
    )
    reader = table.to_reader()
    sol = table.to_pandas()
    t = con.read_in_memory(reader)

    # First execute is fine
    res = t.execute()
    tm.assert_frame_equal(res, sol)

    # Later executes warn
    with pytest.warns(UserWarning, match="RecordBatchReader"):
        t.limit(2).execute()

    # Re-registering over the name with a new reader is fine
    reader = table.to_reader()
    t = con.read_in_memory(reader, table_name=t.get_name())
    res = t.execute()
    tm.assert_frame_equal(res, sol)


def test_csv_with_slash_n_null(con, tmp_path):
    data_path = tmp_path / "data.csv"
    data_path.write_text("a\n1\n3\n\\N\n")
    t = con.read_csv(data_path, nullstr="\\N")
    col = t.a.execute()
    assert pd.isna(col.iat[-1])


@pytest.mark.xfail(
    LINUX and SANDBOXED,
    reason=("nix can't hit GCS because it is sandboxed."),
)
def test_register_filesystem_gcs(con):
    fsspec = pytest.importorskip("fsspec")
    pytest.importorskip("gcsfs")

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


# TODO: Placing the tests for Iceberg read/write here for now.
# Initially, could not find an appropriate folder under ibis/tests/
# with a `con` that implements all the required functions here.
@pytest.mark.parametrize(
    "table_name",
    [
        # "astronauts",
        # "awards_players",
        # "diamonds",
        "functional_alltypes",
    ],
)
def test_pyiceberg(con, tmp_path, table_name):
    # TODO: For initial exploration with pyiceberg only.
    # Will be removed.

    ibis_table = con.table(table_name)
    # out_path = tmp_path / "out.parquet"
    # con.to_parquet(ibis_table, out_path)

    # from ibis.backends.flink.datatypes import get_field_data_types

    # pyflink_table = con._from_ibis_table_to_pyflink_table(ibis_table)
    # pyflink_schema = pyflink_table.get_schema()
    # arrow_schema = create_arrow_schema(
    #     pyflink_schema.get_field_names(), get_field_data_types(pyflink_schema)
    # )

    ## Load catalog
    import pyiceberg
    import pyiceberg.catalog

    catalog = pyiceberg.catalog.load_catalog(
        # name="default",
        **{
            # "type": "rest",
            "uri": "http://localhost:8181",
            "s3.endpoint": "http://localhost:9000",
            # "py-io-impl": "pyiceberg.io.pyarrow.PyArrowFileIO",
            "s3.access-key-id": "admin",
            "s3.secret-access-key": "password",
        }
    )

    namespace_list = catalog.list_namespaces()
    print(f"namespace_list= {namespace_list}")

    ## Create namespace
    namespace = "my_namespace"
    try:
        catalog.create_namespace(namespace)
    except pyiceberg.exceptions.NamespaceAlreadyExistsError:
        pass

    table_list = catalog.list_tables(namespace)
    print(f"table_list= {table_list}")

    ## Create table
    from pyiceberg.schema import Schema
    from pyiceberg.types import (
        BooleanType,
        FloatType,
        DoubleType,
        IntegerType,
        LongType,
        NestedField,
        StringType,
        StructType,
        TimestampType,
    )

    # def functional_alltypes_schema():
    schema = Schema(
        # NestedField(field_id=1, name="datetime", field_type=TimestampType(), required=True),
        # NestedField(field_id=2, name="symbol", field_type=StringType(), required=True),
        # NestedField(field_id=3, name="bid", field_type=FloatType(), required=False),
        # NestedField(field_id=4, name="ask", field_type=DoubleType(), required=False),
        NestedField(field_id=1, name="id", field_type=LongType(), required=False),
        NestedField(field_id=2, name="bool_col", field_type=BooleanType(), required=False),
        # NestedField(field_id=3, name="timestamp_col", field_type=TimestampType(), required=True),
    )

    from pyiceberg.partitioning import PartitionSpec, PartitionField
    from pyiceberg.transforms import DayTransform, IdentityTransform

    # Note: pyiceberg does not support appending to partitioned tables
    partition_spec = PartitionSpec(
        PartitionField(
            # source_id=3, field_id=1000, transform=DayTransform(), name="timestamp_col"
            source_id=1, field_id=1000, transform=IdentityTransform(), name="id"
        )
    )

    from pyiceberg.table.sorting import SortOrder, SortField
    from pyiceberg.transforms import IdentityTransform

    # Sort on the symbol
    sort_order = SortOrder(SortField(source_id=1, transform=IdentityTransform()))

    table_name = "my_table"
    try:
        catalog.create_table(
            # identifier=f"{namespace}.{table_name}",
            # identifier=f"default.{table_name}",
            identifier=(namespace, table_name),
            schema=schema,
            # location="s3://warehouse",
            partition_spec=partition_spec,
            sort_order=sort_order,
        )
    except pyiceberg.exceptions.TableAlreadyExistsError:
        pass

    # import requests.models
    table_list = catalog.list_tables(namespace)
    print(f"table_list= {table_list}")

    ## Load table
    iceberg_table = catalog.load_table(identifier=f"{namespace}.{table_name}")
    print(f"iceberg_table= {iceberg_table}")

    ## Write Iceberg table
    # source_df = ibis_table.to_pandas()
    # df = source_df[["id", "bool_col", "timestamp_col"]]
    pyarrow_df = con.to_pyarrow(ibis_table)
    # pyarrow_df = pyarrow_df.select(["id", "bool_col", "timestamp_col"])
    pyarrow_df = pyarrow_df.select(["id", "bool_col"])
    pyarrow_df = pa.concat_tables([pyarrow_df for _ in range(1000*1000)])

    # iceberg_table.append(pyarrow_df)
    iceberg_table.append(pyarrow_df)
    iceberg_table.overwrite(pyarrow_df)
    # return

    ## Read Iceberg table
    # iceberg_table.scan(
    #     row_filter=GreaterThanOrEqual("trip_distance", 10.0),
    #     selected_fields=("VendorID", "tpep_pickup_datetime", "tpep_dropoff_datetime"),
    # ).to_arrow()
    arrow_table = iceberg_table.scan().to_arrow()
    print(f"arrow_table= {arrow_table}")
    print(f"arrow_table.num_rows= {arrow_table.num_rows}")


@pytest.mark.parametrize(
    "table_name",
    [
        # "astronauts",
        # "awards_players",
        "diamonds",
        # "functional_alltypes",
    ],
)
def test_to_iceberg_and_read_iceberg(con, table_name):
    ibis_table = con.table(table_name)

    # Write table to Iceberg
    catalog_properties = {
        "uri": "http://localhost:8181",
        "s3.endpoint": "http://localhost:9000",
        "s3.access-key-id": "admin",
        "s3.secret-access-key": "password",
    }
    catalog = "my_catalog"
    namespace = "my_namespace"
    ibis_table.to_iceberg(
        catalog_properties=catalog_properties,
        catalog=catalog,
        namespace=namespace,
    )

    # Load the table from Iceberg
    ibis_table_from_iceberg = ibis.read_iceberg(
        table_name=ibis_table.get_name(),
        catalog_properties=catalog_properties,
        catalog=catalog,
        namespace=namespace,
    )
    arrow_table_from_iceberg = ibis_table_from_iceberg.to_pyarrow()
    print(f"arrow_table_from_iceberg= {arrow_table_from_iceberg}")

    arrow_table_from_ibis = ibis_table.to_pyarrow()
    print(f"arrow_table_from_ibis= {arrow_table_from_ibis}")

    assert arrow_table_from_iceberg.num_rows == arrow_table_from_ibis.num_rows
