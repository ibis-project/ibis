from __future__ import annotations

import contextlib
import csv
import gzip
import os
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from pytest import param

import ibis
from ibis.backends.conftest import TEST_TABLES

if TYPE_CHECKING:
    from collections.abc import Iterator

    import pyarrow as pa

pytestmark = pytest.mark.notimpl(["druid", "oracle"])


@contextlib.contextmanager
def pushd(new_dir):
    previous_dir = os.getcwd()
    os.chdir(new_dir)
    try:
        yield
    finally:
        os.chdir(previous_dir)


@pytest.fixture
def gzip_csv(data_dir, tmp_path):
    basename = "diamonds.csv"
    f = tmp_path.joinpath(f"{basename}.gz")
    data = data_dir.joinpath("csv", basename).read_bytes()
    f.write_bytes(gzip.compress(data))
    return str(f.absolute())


@pytest.mark.parametrize(
    ("fname", "in_table_name", "out_table_name"),
    [
        param("diamonds.csv", None, "ibis_read_csv_", id="default"),
        param(
            "csv://diamonds.csv",
            "Diamonds2",
            "Diamonds2",
            id="csv_name",
            marks=pytest.mark.notyet(
                ["pyspark"], reason="pyspark lowercases view names"
            ),
        ),
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
            marks=pytest.mark.notyet(
                ["pyspark"], reason="no spaces allowed in view names"
            ),
        ),
        param(
            ["file://diamonds.csv", "diamonds.csv"],
            "fancy_stones2",
            "fancy_stones2",
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
        "snowflake",
        "sqlite",
        "trino",
    ]
)
def test_register_csv(con, data_dir, fname, in_table_name, out_table_name):
    with pushd(data_dir / "csv"):
        table = con.register(fname, table_name=in_table_name)

    assert any(out_table_name in t for t in con.list_tables())
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
        "snowflake",
        "sqlite",
        "trino",
    ]
)
def test_register_csv_gz(con, data_dir, gzip_csv):
    with pushd(data_dir):
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
        "snowflake",
        "sqlite",
        "trino",
    ]
)
def test_register_with_dotted_name(con, data_dir, tmp_path):
    basename = "foo.bar.baz/diamonds.csv"
    f = tmp_path.joinpath(basename)
    f.parent.mkdir()
    data = data_dir.joinpath("csv", "diamonds.csv").read_bytes()
    f.write_bytes(data)
    table = con.register(str(f.absolute()))

    if con.name != "datafusion":
        table.count().execute()


def read_table(path: Path) -> Iterator[tuple[str, pa.Table]]:
    """For each csv `names` in `data_dir` return a `pyarrow.Table`."""
    pac = pytest.importorskip("pyarrow.csv")

    table_name = path.stem
    schema = TEST_TABLES[table_name]
    convert_options = pac.ConvertOptions(
        column_types={name: typ.to_pyarrow() for name, typ in schema.items()}
    )
    data_dir = path.parent
    return pac.read_csv(data_dir / f"{table_name}.csv", convert_options=convert_options)


@pytest.mark.parametrize(
    ("fname", "in_table_name", "out_table_name"),
    [
        param(
            "parquet://functional_alltypes.parquet", None, "ibis_read_parquet", id="url"
        ),
        param("functional_alltypes.parquet", "funk_all", "funk_all", id="basename"),
        param(
            "parquet://functional_alltypes.parq", "funk_all", "funk_all", id="url_parq"
        ),
        param(
            "parquet://functional_alltypes", None, "ibis_read_parquet", id="url_no_ext"
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
        "snowflake",
        "sqlite",
        "trino",
    ]
)
def test_register_parquet(
    con, tmp_path, data_dir, fname, in_table_name, out_table_name
):
    pq = pytest.importorskip("pyarrow.parquet")

    fname = Path(fname)
    table = read_table(data_dir / "csv" / fname.name)

    pq.write_table(table, tmp_path / fname.name)

    with pushd(tmp_path):
        table = con.register(f"parquet://{fname.name}", table_name=in_table_name)

    assert any(out_table_name in t for t in con.list_tables())

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
    data_dir,
):
    pq = pytest.importorskip("pyarrow.parquet")

    table = read_table(data_dir / "csv" / "functional_alltypes.csv")

    pq.write_table(table, tmp_path / "functional_alltypes.parquet")

    with pushd(tmp_path):
        table = con.register(
            ["parquet://functional_alltypes.parquet", "functional_alltypes.parquet"],
            table_name=None,
        )

    assert any("ibis_read_parquet" in t for t in con.list_tables())
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
    pa = pytest.importorskip("pyarrow")
    pa_t = pa.Table.from_pydict({"x": [1, 2, 3], "y": ["a", "b", "c"]})

    t = con.register(pa_t)
    assert t.x.sum().execute() == 6


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
        "snowflake",
        "sqlite",
        "trino",
    ]
)
def test_csv_reregister_schema(con, tmp_path):
    foo = tmp_path.joinpath("foo.csv")
    with foo.open("w", newline="") as csvfile:
        csv.writer(csvfile, delimiter=",").writerows(
            [
                ["cola", "colb", "colc"],
                [0, 1, 2],
                [1, 5, 6],
                [2, 3.0, "bar"],
            ]
        )

    # For a full file scan, expect correct schema based on final row
    # We also use the same `table_name` for both tests to ensure that
    # the table is re-reflected in sqlalchemy
    foo_table = con.register(foo, table_name="same")
    result_schema = foo_table.schema()

    assert result_schema.names == ("cola", "colb", "colc")
    assert result_schema["cola"].is_integer()
    assert result_schema["colb"].is_float64()
    assert result_schema["colc"].is_string()


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
def test_register_garbage(con, monkeypatch):
    # monkeypatch to avoid downloading extensions in tests
    monkeypatch.setattr(con, "_load_extensions", lambda x: True)

    sa = pytest.importorskip("sqlalchemy")
    with pytest.raises(
        sa.exc.OperationalError, match="No files found that match the pattern"
    ):
        con.read_csv("garbage_notafile")

    with pytest.raises(FileNotFoundError):
        con.read_parquet("garbage_notafile")


@pytest.mark.parametrize(
    ("fname", "in_table_name", "out_table_name"),
    [
        (
            "functional_alltypes.parquet",
            None,
            "ibis_read_parquet",
        ),
        ("functional_alltypes.parquet", "funk_all", "funk_all"),
    ],
)
@pytest.mark.notyet(
    [
        "bigquery",
        "dask",
        "impala",
        "mssql",
        "mysql",
        "pandas",
        "postgres",
        "sqlite",
        "trino",
    ]
)
def test_read_parquet(con, tmp_path, data_dir, fname, in_table_name, out_table_name):
    pq = pytest.importorskip("pyarrow.parquet")

    fname = Path(fname)
    fname = Path(data_dir) / "parquet" / fname.name
    table = pq.read_table(fname)

    pq.write_table(table, tmp_path / fname.name)

    with pushd(data_dir):
        if con.name == "pyspark":
            # pyspark doesn't respect CWD
            fname = str(Path(fname).absolute())
        table = con.read_parquet(fname, table_name=in_table_name)

    assert any(out_table_name in t for t in con.list_tables())

    if con.name != "datafusion":
        table.count().execute()


@pytest.fixture(scope="module")
def ft_data(data_dir):
    pq = pytest.importorskip("pyarrow.parquet")
    nrows = 5
    table = pq.read_table(data_dir.joinpath("parquet", "functional_alltypes.parquet"))
    return table.slice(0, nrows)


@pytest.mark.notyet(
    [
        "bigquery",
        "dask",
        "impala",
        "mssql",
        "mysql",
        "pandas",
        "postgres",
        "sqlite",
        "trino",
    ]
)
def test_read_parquet_glob(con, tmp_path, ft_data):
    pq = pytest.importorskip("pyarrow.parquet")

    nrows = len(ft_data)
    ntables = 2
    ext = "parquet"

    fnames = [f"data{i}.{ext}" for i in range(ntables)]
    for fname in fnames:
        pq.write_table(ft_data, tmp_path / fname)

    table = con.read_parquet(tmp_path / f"*.{ext}")

    assert table.count().execute() == nrows * ntables


@pytest.mark.notyet(
    [
        "bigquery",
        "dask",
        "impala",
        "mssql",
        "mysql",
        "pandas",
        "postgres",
        "sqlite",
        "trino",
    ]
)
def test_read_csv_glob(con, tmp_path, ft_data):
    pc = pytest.importorskip("pyarrow.csv")

    nrows = len(ft_data)
    ntables = 2
    ext = "csv"

    fnames = [f"data{i}.{ext}" for i in range(ntables)]
    for fname in fnames:
        pc.write_csv(ft_data, tmp_path / fname)

    table = con.read_csv(tmp_path / f"*.{ext}")

    assert table.count().execute() == nrows * ntables


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
        "postgres",
        "sqlite",
        "trino",
    ]
)
def test_read_json_glob(con, tmp_path, ft_data):
    nrows = len(ft_data)
    ntables = 2
    ext = "json"

    df = ft_data.to_pandas()

    for i in range(ntables):
        df.to_json(
            tmp_path / f"data{i}.{ext}", orient="records", lines=True, date_format="iso"
        )

    table = con.read_json(tmp_path / f"*.{ext}")

    assert table.count().execute() == nrows * ntables


@pytest.fixture(scope="module")
def num_diamonds(data_dir):
    with open(data_dir / "csv" / "diamonds.csv") as f:
        # subtract 1 for the header
        return sum(1 for _ in f) - 1


DIAMONDS_COLUMN_TYPES = {
    # snowflake's `INFER_SCHEMA` returns this for the diamonds CSV `price`
    # column type
    "snowflake": {
        "carat": "decimal(3, 2)",
        "depth": "decimal(3, 1)",
        "table": "decimal(3, 1)",
        "x": "decimal(4, 2)",
        "y": "decimal(4, 2)",
        "z": "decimal(4, 2)",
    },
    "pyspark": {"price": "int32"},
}


@pytest.mark.parametrize(
    ("in_table_name", "out_table_name"),
    [
        param(None, "ibis_read_csv_", id="default"),
        param("fancy_stones", "fancy_stones", id="file_name"),
    ],
)
@pytest.mark.notyet(
    [
        "bigquery",
        "dask",
        "impala",
        "mssql",
        "mysql",
        "pandas",
        "postgres",
        "sqlite",
        "trino",
    ]
)
def test_read_csv(con, data_dir, in_table_name, out_table_name, num_diamonds):
    fname = "diamonds.csv"
    with pushd(data_dir / "csv"):
        if con.name == "pyspark":
            # pyspark doesn't respect CWD
            fname = str(Path(fname).absolute())
        table = con.read_csv(fname, table_name=in_table_name)

    assert any(out_table_name in t for t in con.list_tables())

    special_types = DIAMONDS_COLUMN_TYPES.get(con.name, {})

    assert table.schema() == ibis.schema(
        {
            "carat": "float64",
            "cut": "string",
            "color": "string",
            "clarity": "string",
            "depth": "float64",
            "table": "float64",
            "price": "int64",
            "x": "float64",
            "y": "float64",
            "z": "float64",
            **special_types,
        }
    )
    assert table.count().execute() == num_diamonds
