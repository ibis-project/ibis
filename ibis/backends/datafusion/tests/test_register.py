from __future__ import annotations

import pandas as pd
import pyarrow as pa
import pytest

import ibis


@pytest.fixture
def conn():
    return ibis.datafusion.connect()


def test_read_csv(conn, data_dir):
    t = conn.read_csv(data_dir / "csv" / "functional_alltypes.csv")
    assert t.count().execute()


def test_read_parquet(conn, data_dir):
    t = conn.read_parquet(data_dir / "parquet" / "functional_alltypes.parquet")
    assert t.count().execute()


def test_register_table(conn):
    tab = pa.table({"x": [1, 2, 3]})
    conn.create_table("my_table", tab)
    assert conn.table("my_table").x.sum().execute() == 6


def test_register_pandas(conn):
    df = pd.DataFrame({"x": [1, 2, 3]})
    conn.create_table("my_table", df)
    assert conn.table("my_table").x.sum().execute() == 6


def test_register_batches(conn):
    batch = pa.record_batch([pa.array([1, 2, 3])], names=["x"])
    conn.create_table("my_table", batch)
    assert conn.table("my_table").x.sum().execute() == 6


def test_register_dataset(conn):
    import pyarrow.dataset as ds

    tab = pa.table({"x": [1, 2, 3]})
    dataset = ds.InMemoryDataset(tab)
    with pytest.warns(FutureWarning, match="v9.1"):
        conn.register(dataset, "my_table")
        assert conn.table("my_table").x.sum().execute() == 6


def test_create_table_with_uppercase_name(conn):
    tab = pa.table({"x": [1, 2, 3]})
    conn.create_table("MY_TABLE", tab)
    assert conn.table("MY_TABLE").x.sum().execute() == 6
