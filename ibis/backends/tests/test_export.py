import sys

import pandas as pd
import pytest
from pytest import param

import ibis

pa = pytest.importorskip("pyarrow")

limit = [
    param(
        42,
        id='limit',
        marks=[
            pytest.mark.notimpl(
                [
                    # limit not implemented for pandas backend execution
                    "dask",
                    "datafusion",
                    "impala",
                    "pandas",
                    "pyspark",
                ]
            ),
        ],
    ),
]

no_limit = [
    param(
        None, id='nolimit', marks=[pytest.mark.notimpl(["dask", "impala", "pyspark"])]
    )
]

limit_no_limit = limit + no_limit


@pytest.mark.parametrize("limit", limit_no_limit)
@pytest.mark.notimpl(["druid"])
def test_table_to_pyarrow_batches(limit, awards_players):
    batch_reader = awards_players.to_pyarrow_batches(limit=limit)
    assert isinstance(batch_reader, pa.ipc.RecordBatchReader)
    batch = batch_reader.read_next_batch()
    assert isinstance(batch, pa.RecordBatch)
    if limit is not None:
        assert len(batch) == limit


@pytest.mark.notyet(
    ["pandas"], reason="DataFrames have no option for outputting in batches"
)
@pytest.mark.parametrize("limit", limit_no_limit)
def test_column_to_pyarrow_batches(limit, awards_players):
    batch_reader = awards_players.awardID.to_pyarrow_batches(limit=limit)
    assert isinstance(batch_reader, pa.ipc.RecordBatchReader)
    batch = batch_reader.read_next_batch()
    assert isinstance(batch, pa.RecordBatch)
    if limit is not None:
        assert len(batch) == limit


@pytest.mark.parametrize("limit", limit_no_limit)
@pytest.mark.notimpl(["druid"])
def test_table_to_pyarrow_table(limit, awards_players):
    table = awards_players.to_pyarrow(limit=limit)
    assert isinstance(table, pa.Table)
    if limit is not None:
        assert len(table) == limit


@pytest.mark.parametrize("limit", limit_no_limit)
def test_column_to_pyarrow_array(limit, awards_players):
    array = awards_players.awardID.to_pyarrow(limit=limit)
    assert isinstance(array, (pa.ChunkedArray, pa.Array))
    if limit is not None:
        assert len(array) == limit


@pytest.mark.parametrize("limit", no_limit)
@pytest.mark.xfail_version(datafusion=["datafusion>=21"])
def test_empty_column_to_pyarrow(limit, awards_players):
    expr = awards_players.filter(awards_players.awardID == "DEADBEEF").awardID
    array = expr.to_pyarrow(limit=limit)
    assert isinstance(array, (pa.ChunkedArray, pa.Array))
    assert len(array) == 0


@pytest.mark.notyet(["datafusion"], reason="DataFusion backend doesn't support sum")
@pytest.mark.parametrize("limit", no_limit)
def test_empty_scalar_to_pyarrow(limit, awards_players):
    expr = awards_players.filter(awards_players.awardID == "DEADBEEF").yearID.sum()
    array = expr.to_pyarrow(limit=limit)
    assert isinstance(array, pa.Scalar)


@pytest.mark.notyet(["datafusion"], reason="DataFusion backend doesn't support sum")
@pytest.mark.parametrize("limit", no_limit)
def test_scalar_to_pyarrow_scalar(limit, awards_players):
    scalar = awards_players.yearID.sum().to_pyarrow(limit=limit)
    assert isinstance(scalar, pa.Scalar)


@pytest.mark.notimpl(["dask", "impala", "pyspark", "druid"])
def test_table_to_pyarrow_table_schema(awards_players):
    table = awards_players.to_pyarrow()
    assert isinstance(table, pa.Table)
    assert table.schema == awards_players.schema().to_pyarrow()


@pytest.mark.notimpl(["dask", "impala", "pyspark"])
def test_column_to_pyarrow_table_schema(awards_players):
    expr = awards_players.awardID
    array = expr.to_pyarrow()
    assert isinstance(array, (pa.ChunkedArray, pa.Array))
    assert array.type == expr.type().to_pyarrow()


@pytest.mark.notimpl(["pandas", "dask", "impala", "pyspark", "datafusion", "druid"])
def test_table_pyarrow_batch_chunk_size(awards_players):
    batch_reader = awards_players.to_pyarrow_batches(limit=2050, chunk_size=2048)
    assert isinstance(batch_reader, pa.ipc.RecordBatchReader)
    batch = batch_reader.read_next_batch()
    assert isinstance(batch, pa.RecordBatch)
    assert len(batch) <= 2048


@pytest.mark.notimpl(["pandas", "dask", "impala", "pyspark", "datafusion"])
def test_column_pyarrow_batch_chunk_size(awards_players):
    batch_reader = awards_players.awardID.to_pyarrow_batches(
        limit=2050, chunk_size=2048
    )
    assert isinstance(batch_reader, pa.ipc.RecordBatchReader)
    batch = batch_reader.read_next_batch()
    assert isinstance(batch, pa.RecordBatch)
    assert len(batch) <= 2048


@pytest.mark.notimpl(["pandas", "dask", "impala", "pyspark", "datafusion", "druid"])
@pytest.mark.broken(
    ["sqlite"],
    raises=pa.ArrowException,
    reason="Test data has empty strings in columns typed as int64",
)
def test_to_pyarrow_batches_borked_types(batting):
    """This is a temporary test to expose an(other) issue with sqlite typing
    shenanigans."""
    batch_reader = batting.to_pyarrow_batches(limit=42)
    assert isinstance(batch_reader, pa.ipc.RecordBatchReader)
    batch = batch_reader.read_next_batch()
    assert isinstance(batch, pa.RecordBatch)
    assert len(batch) == 42


@pytest.mark.notimpl(["dask", "datafusion", "impala", "pyspark"])
def test_to_pyarrow_memtable(con):
    expr = ibis.memtable({"x": [1, 2, 3]})
    table = con.to_pyarrow(expr)
    assert isinstance(table, pa.Table)
    assert len(table) == 3


@pytest.mark.notimpl(["dask", "datafusion", "impala", "pyspark"])
def test_to_pyarrow_batches_memtable(con):
    expr = ibis.memtable({"x": [1, 2, 3]})
    n = 0
    for batch in con.to_pyarrow_batches(expr):
        assert isinstance(batch, pa.RecordBatch)
        n += len(batch)
    assert n == 3


def test_no_pyarrow_message(awards_players, monkeypatch):
    monkeypatch.setitem(sys.modules, "pyarrow", None)
    with pytest.raises(
        ModuleNotFoundError, match="requires `pyarrow` but|import of pyarrow halted"
    ):
        awards_players.to_pyarrow()


@pytest.mark.notimpl(["dask", "impala", "pyspark", "druid"])
def test_table_to_parquet(tmp_path, backend, awards_players):
    outparquet = tmp_path / "out.parquet"
    awards_players.to_parquet(outparquet)

    df = pd.read_parquet(outparquet)

    backend.assert_frame_equal(awards_players.execute(), df)


@pytest.mark.notimpl(
    [
        "bigquery",
        "clickhouse",
        "datafusion",
        "mssql",
        "mysql",
        "pandas",
        "polars",
        "postgres",
        "snowflake",
        "sqlite",
        "trino",
    ],
    reason="no partitioning support",
)
@pytest.mark.notimpl(
    ["dask", "impala", "pyspark", "druid"], reason="No to_parquet support"
)
def test_roundtrip_partitioned_parquet(tmp_path, con, backend, awards_players):
    outparquet = tmp_path / "outhive.parquet"
    awards_players.to_parquet(outparquet, partition_by="yearID")

    assert outparquet.is_dir()

    # Check that the directories are all named as expected
    for d in outparquet.iterdir():
        assert d.stem.startswith("yearID")

    # Reingest and compare schema
    reingest = con.read_parquet(outparquet / "*" / "*")

    assert reingest.schema() == awards_players.schema()

    backend.assert_frame_equal(awards_players.execute(), awards_players.execute())


@pytest.mark.notimpl(["dask", "impala", "pyspark"])
def test_table_to_csv(tmp_path, backend, awards_players):
    outcsv = tmp_path / "out.csv"

    # avoid pandas NaNonense
    awards_players = awards_players.select("playerID", "awardID", "yearID", "lgID")

    awards_players.to_csv(outcsv)

    df = pd.read_csv(outcsv, dtype=awards_players.schema().to_pandas())

    backend.assert_frame_equal(awards_players.execute(), df)
