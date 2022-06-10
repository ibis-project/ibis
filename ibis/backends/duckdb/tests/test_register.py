import contextlib
import os

import pytest

import ibis


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
        ("parquet://batting.parquet", None, "batting"),
        ("batting.parquet", "baseball", "baseball"),
    ],
)
def test_register_file(data_directory, fname, in_table_name, out_table_name):
    con = ibis.duckdb.connect()
    with pushd(data_directory):
        con.register(fname, table_name=in_table_name)

    assert out_table_name in con.list_tables()

    table = con.table(out_table_name)
    assert table.count().execute() > 0
