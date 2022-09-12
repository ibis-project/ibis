import contextlib
import gzip
import os
from pathlib import Path

import pytest
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
    "fname, in_table_name, out_table_name",
    [
        param("diamonds.csv", None, "diamonds", id="default"),
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
    if ext is not None:
        fname = gzip_csv
    with pushd(data_directory):
        con.register(fname, table_name=in_table_name)

    assert out_table_name in con.list_tables()

    table = con.table(out_table_name)
    assert table.count().execute()


def test_register_with_dotted_name(data_directory, tmp_path):
    con = ibis.duckdb.connect()
    basename = "foo.bar.baz/diamonds.csv"
    f = tmp_path.joinpath(basename)
    f.parent.mkdir()
    data = data_directory.joinpath("diamonds.csv").read_bytes()
    f.write_bytes(data)
    con.register(str(f.absolute()))
    table = con.table("diamonds")
    assert table.count().execute()


@pytest.mark.parametrize(
    "fname, in_table_name, out_table_name",
    [
        (
            "parquet://functional_alltypes.parquet",
            None,
            "functional_alltypes",
        ),
        ("functional_alltypes.parquet", "funk_all", "funk_all"),
        ("parquet://functional_alltypes.parq", "funk_all", "funk_all"),
        ("parquet://functional_alltypes", None, "functional_alltypes"),
    ],
)
def test_register_parquet(
    tmp_path, data_directory, fname, in_table_name, out_table_name
):
    pq = pytest.importorskip("pyarrow.parquet")

    fname = Path(fname)
    _, table = next(read_tables([fname.stem], data_directory))

    pq.write_table(table, tmp_path / fname.name)

    con = ibis.duckdb.connect()
    with pushd(tmp_path):
        con.register(f"parquet://{fname.name}", table_name=in_table_name)

    assert out_table_name in con.list_tables()

    table = con.table(out_table_name)
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
    pa = pytest.importorskip("pyarrow")
    pa_t = pa.Table.from_pydict({"x": [1, 2, 3], "y": ["a", "b", "c"]})

    con = ibis.duckdb.connect()

    t = con.register(pa_t)
    assert t.x.sum().execute() == 6
