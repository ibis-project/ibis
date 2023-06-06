import pandas as pd
import pandas.testing as tm
import pytest
import sqlalchemy as sa
from packaging.version import parse as vparse
from pytest import param

import ibis
import ibis.expr.datatypes as dt
from ibis import util

pa = pytest.importorskip("pyarrow")

try:
    from pyspark.sql.utils import ParseException
except ImportError:
    ParseException = None

try:
    from deltalake import PyDeltaTableError
except ImportError:
    PyDeltaTableError = None

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
    util.consume(batch_reader)


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
    util.consume(batch_reader)


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


@pytest.mark.parametrize("limit", no_limit)
def test_empty_scalar_to_pyarrow(limit, awards_players):
    expr = awards_players.filter(awards_players.awardID == "DEADBEEF").yearID.sum()
    array = expr.to_pyarrow(limit=limit)
    assert isinstance(array, pa.Scalar)


@pytest.mark.parametrize("limit", no_limit)
def test_scalar_to_pyarrow_scalar(limit, awards_players):
    scalar = awards_players.yearID.sum().to_pyarrow(limit=limit)
    assert isinstance(scalar, pa.Scalar)


@pytest.mark.notimpl(["dask", "impala", "pyspark", "druid"])
def test_table_to_pyarrow_table_schema(con, awards_players):
    table = awards_players.to_pyarrow()
    assert isinstance(table, pa.Table)

    string = (
        pa.large_string()
        if con.name == "duckdb" and vparse(con.version) >= vparse("0.8.0")
        else pa.string()
    )
    expected_schema = pa.schema(
        [
            pa.field("playerID", string),
            pa.field("awardID", string),
            pa.field("yearID", pa.int64()),
            pa.field("lgID", string),
            pa.field("tie", string),
            pa.field("notes", string),
        ]
    )
    assert table.schema == expected_schema


@pytest.mark.notimpl(["dask", "impala", "pyspark"])
def test_column_to_pyarrow_table_schema(awards_players):
    expr = awards_players.awardID
    array = expr.to_pyarrow()
    assert isinstance(array, (pa.ChunkedArray, pa.Array))
    assert array.type == pa.string() or array.type == pa.large_string()


@pytest.mark.notimpl(["pandas", "dask", "impala", "pyspark", "datafusion", "druid"])
@pytest.mark.notyet(
    ["clickhouse"],
    raises=AssertionError,
    reason="clickhouse connect doesn't seem to respect `max_block_size` parameter",
)
def test_table_pyarrow_batch_chunk_size(awards_players):
    batch_reader = awards_players.to_pyarrow_batches(limit=2050, chunk_size=2048)
    assert isinstance(batch_reader, pa.ipc.RecordBatchReader)
    batch = batch_reader.read_next_batch()
    assert isinstance(batch, pa.RecordBatch)
    assert len(batch) <= 2048
    util.consume(batch_reader)


@pytest.mark.notimpl(["pandas", "dask", "impala", "pyspark", "datafusion"])
@pytest.mark.notyet(
    ["clickhouse"],
    raises=AssertionError,
    reason="clickhouse connect doesn't seem to respect `max_block_size` parameter",
)
def test_column_pyarrow_batch_chunk_size(awards_players):
    batch_reader = awards_players.awardID.to_pyarrow_batches(
        limit=2050, chunk_size=2048
    )
    assert isinstance(batch_reader, pa.ipc.RecordBatchReader)
    batch = batch_reader.read_next_batch()
    assert isinstance(batch, pa.RecordBatch)
    assert len(batch) <= 2048
    util.consume(batch_reader)


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
    util.consume(batch_reader)


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
        "oracle",
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

    # avoid type comparison to appease duckdb: as of 0.8.0 it returns large_string
    assert reingest.schema().names == awards_players.schema().names

    backend.assert_frame_equal(awards_players.execute(), awards_players.execute())


@pytest.mark.notimpl(["dask", "impala", "pyspark"])
def test_table_to_csv(tmp_path, backend, awards_players):
    outcsv = tmp_path / "out.csv"

    # avoid pandas NaNonense
    awards_players = awards_players.select("playerID", "awardID", "yearID", "lgID")

    awards_players.to_csv(outcsv)

    df = pd.read_csv(outcsv, dtype=awards_players.schema().to_pandas())

    backend.assert_frame_equal(awards_players.execute(), df)


@pytest.mark.parametrize(
    ("dtype", "pyarrow_dtype"),
    [
        param(
            dt.Decimal(38, 9),
            pa.Decimal128Type,
            id="decimal128",
            marks=[
                pytest.mark.broken(
                    ["impala"], raises=AttributeError, reason="fetchmany doesn't exist"
                ),
                pytest.mark.notyet(["druid"], raises=sa.exc.ProgrammingError),
                pytest.mark.notyet(["dask"], raises=NotImplementedError),
                pytest.mark.notyet(["pyspark"], raises=NotImplementedError),
            ],
        ),
        param(
            dt.Decimal(76, 38),
            pa.Decimal256Type,
            id="decimal256",
            marks=[
                pytest.mark.broken(["pandas"], raises=AssertionError),
                pytest.mark.notyet(["impala"], reason="precision not supported"),
                pytest.mark.notyet(
                    ["druid", "duckdb", "snowflake", "trino"],
                    raises=sa.exc.ProgrammingError,
                ),
                pytest.mark.notyet(["oracle"], raises=sa.exc.DatabaseError),
                pytest.mark.notyet(["dask"], raises=NotImplementedError),
                pytest.mark.notyet(["mssql", "mysql"], raises=sa.exc.OperationalError),
                pytest.mark.notyet(["pyspark"], raises=ParseException),
            ],
        ),
    ],
)
def test_to_pyarrow_decimal(backend, dtype, pyarrow_dtype):
    result = (
        backend.functional_alltypes.limit(1)
        .double_col.cast(dtype)
        .name("dec")
        .to_pyarrow()
    )
    assert len(result) == 1
    assert isinstance(result.type, pyarrow_dtype)


@pytest.mark.notyet(
    [
        "bigquery",
        "clickhouse",
        "datafusion",
        "impala",
        "mssql",
        "mysql",
        "oracle",
        "postgres",
        "pyspark",
        "snowflake",
        "sqlite",
        "trino",
    ],
    raises=AttributeError,
    reason="read_delta not yet implemented",
)
@pytest.mark.notyet(["pandas"], raises=PyDeltaTableError)
@pytest.mark.notyet(["dask"], raises=NotImplementedError)
@pytest.mark.notyet(
    ["druid"],
    raises=pa.lib.ArrowTypeError,
    reason="arrow type conversion fails in `to_delta` call",
)
def test_roundtrip_delta(con, alltypes, tmp_path, monkeypatch):
    pytest.importorskip("deltalake")

    t = alltypes.head()
    expected = t.execute()
    path = tmp_path / "test.delta"
    t.to_delta(path)

    monkeypatch.setattr(ibis.options, "default_backend", con)
    dt = ibis.read_delta(path)
    result = dt.to_pandas()

    tm.assert_frame_equal(result, expected)
