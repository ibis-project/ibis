import contextlib
import csv
import gzip
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import sqlalchemy as sa
from pytest import param

import ibis
from ibis.backends.conftest import read_tables


@contextlib.contextmanager
def pushd(new_dir):
    previous_dir = os.getcwd()
    os.chdir(new_dir)
    try:
        yield
    finally:
        os.chdir(previous_dir)


@pytest.fixture
def gzip_csv(data_directory, tmp_path):
    basename = "diamonds.csv"
    f = tmp_path.joinpath(f"{basename}.gz")
    data = data_directory.joinpath(basename).read_bytes()
    f.write_bytes(gzip.compress(data))
    return str(f.absolute())


@pytest.mark.parametrize(
    ("fname", "in_table_name", "out_table_name"),
    [
        param("diamonds.csv", None, "ibis_read_csv_", id="default"),
        param("csv://diamonds.csv", "Diamonds2", "Diamonds2", id="csv_name"),
        param(
            "file://diamonds.csv",
            "fancy_stones",
            "fancy_stones",
            id="file_name",
        ),
        param(
            "file://diamonds.csv",
            "fancy stones",
            "fancy stones",
            id="file_atypical_name",
        ),
        param(
            ["file://diamonds.csv", "diamonds.csv"],
            "fancy stones",
            "fancy stones",
            id="multi_csv",
            marks=pytest.mark.notyet(
                ["polars", "datafusion"],
                reason="doesn't accept multiple files to scan or read",
            ),
        ),
    ],
)
@pytest.mark.notyet(
    [
        "bigquery",
        "clickhouse",
        "dask",
        "impala",
        "mssql",
        "mysql",
        "pandas",
        "postgres",
        "pyspark",
        "snowflake",
        "sqlite",
        "trino",
    ]
)
def test_register_csv(con, data_directory, fname, in_table_name, out_table_name):
    with pushd(data_directory):
        table = con.register(fname, table_name=in_table_name)

    assert any(t.startswith(out_table_name) for t in con.list_tables())
    if con.name != "datafusion":
        table.count().execute()


@pytest.mark.notimpl(["datafusion"])
@pytest.mark.notyet(
    [
        "bigquery",
        "clickhouse",
        "dask",
        "impala",
        "mssql",
        "mysql",
        "pandas",
        "postgres",
        "pyspark",
        "snowflake",
        "sqlite",
        "trino",
    ]
)
def test_register_csv_gz(con, data_directory, gzip_csv):
    with pushd(data_directory):
        table = con.register(gzip_csv)

    assert table.count().execute()


@pytest.mark.notyet(
    [
        "bigquery",
        "clickhouse",
        "dask",
        "impala",
        "mssql",
        "mysql",
        "pandas",
        "postgres",
        "pyspark",
        "snowflake",
        "sqlite",
        "trino",
    ]
)
def test_register_with_dotted_name(con, data_directory, tmp_path):
    basename = "foo.bar.baz/diamonds.csv"
    f = tmp_path.joinpath(basename)
    f.parent.mkdir()
    data = data_directory.joinpath("diamonds.csv").read_bytes()
    f.write_bytes(data)
    table = con.register(str(f.absolute()))

    if con.name != "datafusion":
        table.count().execute()


@pytest.mark.parametrize(
    ("fname", "in_table_name", "out_table_name"),
    [
        pytest.param(
            "parquet://functional_alltypes.parquet",
            None,
            "ibis_read_parquet",
        ),
        ("functional_alltypes.parquet", "funk_all", "funk_all"),
        pytest.param("parquet://functional_alltypes.parq", "funk_all", "funk_all"),
        ("parquet://functional_alltypes", None, "ibis_read_parquet"),
    ],
)
@pytest.mark.notyet(
    [
        "bigquery",
        "clickhouse",
        "dask",
        "impala",
        "mssql",
        "mysql",
        "pandas",
        "postgres",
        "pyspark",
        "snowflake",
        "sqlite",
        "trino",
    ]
)
def test_register_parquet(
    con, tmp_path, data_directory, fname, in_table_name, out_table_name
):
    fname = Path(fname)
    _, table = next(read_tables([fname.stem], data_directory))

    pq.write_table(table, tmp_path / fname.name)

    with pushd(tmp_path):
        table = con.register(f"parquet://{fname.name}", table_name=in_table_name)

    assert any(t.startswith(out_table_name) for t in con.list_tables())

    if con.name != "datafusion":
        table.count().execute()


@pytest.mark.notyet(
    [
        "bigquery",
        "clickhouse",
        "dask",
        "datafusion",
        "impala",
        "mssql",
        "mysql",
        "pandas",
        "polars",  # polars supports parquet dirs, not lists of files
        "postgres",
        "pyspark",
        "snowflake",
        "sqlite",
        "trino",
    ]
)
def test_register_iterator_parquet(
    con,
    tmp_path,
    data_directory,
):
    pq = pytest.importorskip("pyarrow.parquet")

    _, table = next(read_tables(["functional_alltypes"], data_directory))

    pq.write_table(table, tmp_path / "functional_alltypes.parquet")

    with pushd(tmp_path):
        table = con.register(
            ["parquet://functional_alltypes.parquet", "functional_alltypes.parquet"],
            table_name=None,
        )

    assert any(t.startswith("ibis_read_parquet") for t in con.list_tables())

    assert table.count().execute()


@pytest.mark.notimpl(["datafusion"])
@pytest.mark.notyet(
    [
        "bigquery",
        "clickhouse",
        "dask",
        "impala",
        "mssql",
        "mysql",
        "pandas",
        "postgres",
        "pyspark",
        "snowflake",
        "sqlite",
        "trino",
    ]
)
def test_register_pandas(con):
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})

    t = con.register(df)
    assert t.x.sum().execute() == 6

    t = con.register(df, "my_table")
    assert t.op().name == "my_table"
    assert t.x.sum().execute() == 6


@pytest.mark.notimpl(["datafusion", "polars"])
@pytest.mark.notyet(
    [
        "bigquery",
        "clickhouse",
        "dask",
        "impala",
        "mssql",
        "mysql",
        "pandas",
        "postgres",
        "pyspark",
        "snowflake",
        "sqlite",
        "trino",
    ]
)
def test_register_pyarrow_tables(con):
    pa_t = pa.Table.from_pydict({"x": [1, 2, 3], "y": ["a", "b", "c"]})

    t = con.register(pa_t)
    assert t.x.sum().execute() == 6


@pytest.mark.broken(
    ["polars"], reason="it's working but it infers the int column as 32"
)
@pytest.mark.notimpl(["datafusion"])
@pytest.mark.notyet(
    [
        "bigquery",
        "clickhouse",
        "dask",
        "impala",
        "mssql",
        "mysql",
        "pandas",
        "postgres",
        "pyspark",
        "snowflake",
        "sqlite",
        "trino",
    ]
)
def test_csv_reregister_schema(con, tmp_path):

    foo = tmp_path / "foo.csv"
    with open(foo, "w", newline="") as csvfile:
        foowriter = csv.writer(
            csvfile,
            delimiter=",",
        )
        foowriter.writerow(["cola", "colb", "colc"])
        foowriter.writerow([0, 1, 2])
        foowriter.writerow([1, 5, 6])
        foowriter.writerow([2, 3.0, "bar"])

    # For a full file scan, expect correct schema based on final row
    foo_table = con.register(foo)
    exp_schema = ibis.schema(dict(cola="int32", colb="float64", colc="string"))
    assert foo_table.schema() == exp_schema

    # If file scan is limited to first two rows, should be all int32
    foo_table = con.register(foo, SAMPLE_SIZE=2)
    exp_schema = ibis.schema(dict(cola="int32", colb="int32", colc="int32"))
    assert foo_table.schema() == exp_schema


@pytest.mark.notimpl(
    [
        "bigquery",
        "clickhouse",
        "dask",
        "datafusion",
        "impala",
        "mysql",
        "mssql",
        "pandas",
        "polars",
        "postgres",
        "pyspark",
        "snowflake",
        "sqlite",
        "trino",
    ]
)
def test_register_garbage(con):
    with pytest.raises(
        sa.exc.OperationalError, match="No files found that match the pattern"
    ):
        con.read_csv("garbage_notafile")

    with pytest.raises(
        sa.exc.OperationalError, match="No files found that match the pattern"
    ):
        con.read_parquet("garbage_notafile")
