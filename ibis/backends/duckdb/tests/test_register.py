import contextlib
import os
from pathlib import Path

import pytest

import ibis
from ibis.backends.conftest import read_tables


@contextlib.contextmanager
def pushd(new_dir):
    previous_dir = os.getcwd()
    os.chdir(new_dir)
    yield
    os.chdir(previous_dir)


@pytest.mark.parametrize(
    "fname, in_table_name, out_table_name",
    [
        ("diamonds.csv", None, "diamonds"),
        ("csv://diamonds.csv", "Diamonds", "Diamonds"),
    ],
)
def test_register_csv(data_directory, fname, in_table_name, out_table_name):
    con = ibis.duckdb.connect()
    with pushd(data_directory):
        con.register(fname, table_name=in_table_name)

    assert out_table_name in con.list_tables()

    table = con.table(out_table_name)
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
    import pyarrow.parquet as pq

    fname = Path(fname)
    _, table = next(read_tables([fname.stem], data_directory))

    pq.write_table(table, tmp_path / fname.name)

    con = ibis.duckdb.connect()
    with pushd(tmp_path):
        con.register("parquet://" + str(fname.name), table_name=in_table_name)

    assert out_table_name in con.list_tables()

    table = con.table(out_table_name)
    assert table.count().execute()
