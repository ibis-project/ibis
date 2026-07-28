"""Arrow / pandas / polars output paths, including real streaming."""

from __future__ import annotations

import pyarrow as pa
import pytest

import ibis
from ibis import _


def test_to_pyarrow(mem):
    t = ibis.memtable({"x": [1, 2, 3], "s": ["a", "b", "c"]})
    tbl = mem.to_pyarrow(t.order_by("x"))
    assert isinstance(tbl, pa.Table)
    assert tbl.column("x").to_pylist() == [1, 2, 3]
    assert tbl.column("s").to_pylist() == ["a", "b", "c"]


def test_to_pyarrow_scalar(mem):
    t = ibis.memtable({"x": [1, 2, 3]})
    assert mem.to_pyarrow(t.x.sum()).as_py() == 6


def test_to_pyarrow_batches_returns_reader(mem):
    t = mem.sql("SELECT number AS n FROM numbers(10)")
    reader = mem.to_pyarrow_batches(t)
    assert isinstance(reader, pa.ipc.RecordBatchReader)
    assert reader.read_all().num_rows == 10


def test_to_pyarrow_batches_multiple_batches(mem):
    # A result larger than chDB's internal block size streams as more than one
    # batch, exercising the real (non-materializing) record_batch path.
    n = 200_000
    t = mem.sql(f"SELECT number AS n FROM numbers({n})")
    reader = mem.to_pyarrow_batches(t, chunk_size=65_536)
    batches = list(reader)
    assert len(batches) > 1
    assert sum(b.num_rows for b in batches) == n


def test_to_pyarrow_batches_schema_matches(mem):
    t = mem.sql("SELECT number AS n, toString(number) AS s FROM numbers(3)")
    reader = mem.to_pyarrow_batches(t)
    assert reader.schema.names == ["n", "s"]


def test_to_pandas(mem):
    t = ibis.memtable({"x": [1, 2, 3]})
    df = mem.execute(t.order_by("x"))
    assert df["x"].tolist() == [1, 2, 3]


def test_to_polars(mem):
    pl = pytest.importorskip("polars")
    t = ibis.memtable({"x": [1, 2, 3], "s": ["a", "b", "c"]})
    df = mem.to_polars(t.order_by("x"))
    assert isinstance(df, pl.DataFrame)
    assert df["x"].to_list() == [1, 2, 3]


def test_empty_result_preserves_schema(mem):
    t = mem.sql("SELECT number AS n FROM numbers(10)").filter(_.n > 100)
    tbl = mem.to_pyarrow(t)
    assert tbl.num_rows == 0
    assert tbl.schema.names == ["n"]
