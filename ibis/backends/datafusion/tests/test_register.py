from __future__ import annotations

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
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
    conn.register(tab, "my_table")
    assert conn.table("my_table").x.sum().execute() == 6


def test_register_pandas(conn):
    df = pd.DataFrame({"x": [1, 2, 3]})
    conn.register(df, "my_table")
    assert conn.table("my_table").x.sum().execute() == 6


def test_register_batches(conn):
    batch = pa.record_batch([pa.array([1, 2, 3])], names=["x"])
    conn.register(batch, "my_table")
    assert conn.table("my_table").x.sum().execute() == 6


def test_register_dataset(conn):
    tab = pa.table({"x": [1, 2, 3]})
    dataset = ds.InMemoryDataset(tab)
    conn.register(dataset, "my_table")
    assert conn.table("my_table").x.sum().execute() == 6
