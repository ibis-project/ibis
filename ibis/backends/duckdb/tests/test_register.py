import contextlib
import csv
import gzip
import os
import platform
import re
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from packaging.version import parse as vparse
from pytest import param

import ibis
from ibis.backends.conftest import read_tables
from ibis.backends.duckdb import _generate_view_code


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
        param("diamonds.csv", None, None, id="default"),
        param("csv://diamonds.csv", "Diamonds", "Diamonds", id="csv_name"),
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
    ],
)
@pytest.mark.parametrize("ext", [None, "csv.gz"])
def test_register_csv(
    data_directory, fname, in_table_name, out_table_name, ext, gzip_csv
):
    con = ibis.duckdb.connect()
    if ext:
        fname = gzip_csv
    with pushd(data_directory):
        table = con.register(fname, table_name=in_table_name)

    if out_table_name is not None:
        out_table_name += (os.extsep * bool(ext) + (ext or "")) * (
            in_table_name is None
        )
        assert out_table_name in con.list_tables()

    assert table.count().execute()


def test_register_with_dotted_name(data_directory, tmp_path):
    con = ibis.duckdb.connect()
    basename = "foo.bar.baz/diamonds.csv"
    f = tmp_path.joinpath(basename)
    f.parent.mkdir()
    data = data_directory.joinpath("diamonds.csv").read_bytes()
    f.write_bytes(data)
    table = con.register(str(f.absolute()))
    assert table.count().execute()


@pytest.mark.parametrize(
    ("fname", "in_table_name", "out_table_name"),
    [
        pytest.param(
            "parquet://functional_alltypes.parquet",
            None,
            "functional_alltypes_parquet",
        ),
        ("functional_alltypes.parquet", "funk_all", "funk_all"),
        ("parquet://functional_alltypes.parq", "funk_all", "funk_all"),
        ("parquet://functional_alltypes", None, "functional_alltypes"),
    ],
)
def test_register_parquet(
    tmp_path, data_directory, fname, in_table_name, out_table_name
):
    fname = Path(fname)
    _, table = next(read_tables([fname.stem], data_directory))

    pq.write_table(table, tmp_path / fname.name)

    con = ibis.duckdb.connect()
    with pushd(tmp_path):
        table = con.register(f"parquet://{fname.name}", table_name=in_table_name)

    assert any(out_table_name in t for t in con.list_tables())

    assert table.count().execute()


def test_register_pandas():
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})

    con = ibis.duckdb.connect()

    t = con.register(df)
    assert t.x.sum().execute() == 6

    t = con.register(df, "my_table")
    assert t.op().name == "my_table"
    assert t.x.sum().execute() == 6


def test_register_pyarrow_tables():
    pa_t = pa.Table.from_pydict({"x": [1, 2, 3], "y": ["a", "b", "c"]})

    con = ibis.duckdb.connect()

    t = con.register(pa_t)
    assert t.x.sum().execute() == 6


@pytest.mark.parametrize(
    "kwargs, expected_snippet",
    [({}, "auto_detect=True"), ({"columns": {"foo": "int8"}}, "auto_detect=False")],
)
def test_csv_register_kwargs(kwargs, expected_snippet):
    view_str, _, _ = _generate_view_code("bork.csv", **kwargs)
    assert expected_snippet in view_str


def test_csv_reregister_schema(tmp_path):
    con = ibis.duckdb.connect()

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


def test_read_csv(data_directory):
    t = ibis.read(data_directory / "functional_alltypes.csv")
    assert t.count().execute()


def test_read_parquet(data_directory):
    t = ibis.read(data_directory / "functional_alltypes.parquet")
    assert t.count().execute()


@pytest.mark.parametrize("basename", ["functional_alltypes.*", "df.xlsx"])
def test_read_invalid(data_directory, basename):
    path = data_directory / basename
    msg = f"^Unrecognized file type or extension: {re.escape(str(path))}"
    with pytest.raises(ValueError, match=msg):
        ibis.read(path)


def test_temp_directory(tmp_path):
    query = "SELECT value FROM duckdb_settings() WHERE name = 'temp_directory'"

    # 1. in-memory + no temp_directory specified
    con = ibis.duckdb.connect()
    [(value,)] = con.con.execute(query).fetchall()
    assert value  # we don't care what the specific value is

    temp_directory = Path(tempfile.gettempdir()) / "duckdb"

    # 2. in-memory + temp_directory specified
    con = ibis.duckdb.connect(temp_directory=temp_directory)
    [(value,)] = con.con.execute(query).fetchall()
    assert value == str(temp_directory)

    # 3. on-disk + no temp_directory specified
    # untested, duckdb sets the temp_directory to something implementation
    # defined

    # 4. on-disk + temp_directory specified
    con = ibis.duckdb.connect(tmp_path / "test2.ddb", temp_directory=temp_directory)
    [(value,)] = con.con.execute(query).fetchall()
    assert value == str(temp_directory)


@pytest.mark.parametrize(
    "path", ["s3://data-lake/dataset/", "s3://data-lake/dataset/file_1.parquet"]
)
@pytest.mark.xfail(
    platform.system() == "Darwin" and vparse(pa.__version__) < vparse("9"),
    reason="pyarrow < 9 macos wheels not built with S3 support",
    raises=pa.ArrowNotImplementedError,
)
def test_s3_parquet(path):
    with pytest.raises(OSError):
        _generate_view_code(path)


@pytest.mark.parametrize("scheme", ["postgres", "postgresql"])
@pytest.mark.parametrize(
    "name, quoted_name", [("test", "test"), ("my table", '"my table"')]
)
def test_postgres(scheme, name, quoted_name):
    uri = f"{scheme}://username:password@localhost:5432"
    sql, table_name, exts = _generate_view_code(uri, name)
    assert sql == (
        f"CREATE OR REPLACE VIEW {quoted_name} AS "
        f"SELECT * FROM postgres_scan_pushdown('{uri}', 'public', '{name}')"
    )
    assert table_name == name
    assert exts == ["postgres_scanner"]
