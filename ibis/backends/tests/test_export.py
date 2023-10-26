from __future__ import annotations

import pandas as pd
import pandas.testing as tm
import pyarrow as pa
import pytest
import sqlalchemy as sa
from pytest import param

import ibis
import ibis.expr.datatypes as dt
from ibis import util
from ibis.formats.pyarrow import PyArrowType

try:
    from pyspark.sql.utils import AnalysisException
except ImportError:
    AnalysisException = None

try:
    from deltalake import PyDeltaTableError
except ImportError:
    PyDeltaTableError = None


limit = [
    param(
        42,
        id="limit",
        marks=[
            pytest.mark.notimpl(
                [
                    # limit not implemented for pandas backend execution
                    "dask",
                    "datafusion",
                    "pandas",
                    "pyspark",
                ]
            ),
        ],
    ),
]

no_limit = [param(None, id="nolimit")]

limit_no_limit = limit + no_limit


@pytest.mark.parametrize("limit", limit_no_limit)
def test_table_to_pyarrow_batches(limit, awards_players):
    with awards_players.to_pyarrow_batches(limit=limit) as batch_reader:
        assert isinstance(batch_reader, pa.ipc.RecordBatchReader)
        batch = batch_reader.read_next_batch()
        assert isinstance(batch, pa.RecordBatch)
        if limit is not None:
            assert len(batch) == limit
        util.consume(batch_reader)


@pytest.mark.parametrize("limit", limit_no_limit)
def test_column_to_pyarrow_batches(limit, awards_players):
    with awards_players.awardID.to_pyarrow_batches(limit=limit) as batch_reader:
        assert isinstance(batch_reader, pa.ipc.RecordBatchReader)
        batch = batch_reader.read_next_batch()
        assert isinstance(batch, pa.RecordBatch)
        if limit is not None:
            assert len(batch) == limit
        util.consume(batch_reader)


@pytest.mark.parametrize("limit", limit_no_limit)
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


@pytest.mark.notimpl(["druid"])
def test_table_to_pyarrow_table_schema(awards_players):
    table = awards_players.to_pyarrow()
    assert isinstance(table, pa.Table)

    string = pa.string()
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


def test_column_to_pyarrow_table_schema(awards_players):
    expr = awards_players.awardID
    array = expr.to_pyarrow()
    assert isinstance(array, (pa.ChunkedArray, pa.Array))
    assert array.type == pa.string() or array.type == pa.large_string()


@pytest.mark.notimpl(["pandas", "dask", "datafusion"])
@pytest.mark.notyet(
    ["clickhouse"],
    raises=AssertionError,
    reason="clickhouse connect doesn't seem to respect `max_block_size` parameter",
)
def test_table_pyarrow_batch_chunk_size(awards_players):
    with awards_players.to_pyarrow_batches(limit=2050, chunk_size=2048) as batch_reader:
        assert isinstance(batch_reader, pa.ipc.RecordBatchReader)
        batch = batch_reader.read_next_batch()
        assert isinstance(batch, pa.RecordBatch)
        assert len(batch) <= 2048
        util.consume(batch_reader)


@pytest.mark.notimpl(["pandas", "dask", "datafusion"])
@pytest.mark.notyet(
    ["clickhouse"],
    raises=AssertionError,
    reason="clickhouse connect doesn't seem to respect `max_block_size` parameter",
)
def test_column_pyarrow_batch_chunk_size(awards_players):
    with awards_players.awardID.to_pyarrow_batches(
        limit=2050, chunk_size=2048
    ) as batch_reader:
        assert isinstance(batch_reader, pa.ipc.RecordBatchReader)
        batch = batch_reader.read_next_batch()
        assert isinstance(batch, pa.RecordBatch)
        assert len(batch) <= 2048
        util.consume(batch_reader)


@pytest.mark.notimpl(["pandas", "dask", "datafusion"])
@pytest.mark.broken(
    ["pyspark"], raises=AssertionError, reason="chunk_size isn't respected"
)
@pytest.mark.broken(
    ["sqlite"],
    raises=pa.ArrowException,
    reason="Test data has empty strings in columns typed as int64",
)
def test_to_pyarrow_batches_borked_types(batting):
    """This is a temporary test to expose an(other) issue with sqlite typing
    shenanigans."""
    with batting.to_pyarrow_batches(limit=42) as batch_reader:
        assert isinstance(batch_reader, pa.ipc.RecordBatchReader)
        batch = batch_reader.read_next_batch()
        assert isinstance(batch, pa.RecordBatch)
        assert len(batch) == 42
        util.consume(batch_reader)


def test_to_pyarrow_memtable(con):
    expr = ibis.memtable({"x": [1, 2, 3]})
    table = con.to_pyarrow(expr)
    assert isinstance(table, pa.Table)
    assert len(table) == 3


def test_to_pyarrow_batches_memtable(con):
    expr = ibis.memtable({"x": [1, 2, 3]})
    n = 0
    with con.to_pyarrow_batches(expr) as batch_reader:
        for batch in batch_reader:
            assert isinstance(batch, pa.RecordBatch)
            n += len(batch)
    assert n == 3


def test_table_to_parquet(tmp_path, backend, awards_players):
    outparquet = tmp_path / "out.parquet"
    awards_players.to_parquet(outparquet)

    df = pd.read_parquet(outparquet)

    backend.assert_frame_equal(awards_players.to_pandas(), df)


@pytest.mark.notimpl(
    [
        "bigquery",
        "clickhouse",
        "dask",
        "datafusion",
        "impala",
        "mssql",
        "mysql",
        "oracle",
        "pandas",
        "polars",
        "postgres",
        "pyspark",
        "snowflake",
        "sqlite",
        "trino",
    ],
    reason="no partitioning support",
)
@pytest.mark.notimpl(["druid"], reason="No to_parquet support")
def test_roundtrip_partitioned_parquet(tmp_path, con, backend, awards_players):
    outparquet = tmp_path / "outhive.parquet"
    awards_players.to_parquet(outparquet, partition_by="yearID")

    assert outparquet.is_dir()

    # Check that the directories are all named as expected
    for d in outparquet.iterdir():
        assert d.stem.startswith("yearID")

    # Reingest and compare schema
    reingest = con.read_parquet(outparquet / "*" / "*")
    reingest = reingest.cast({"yearID": "int64"})

    # avoid type comparison to appease duckdb: as of 0.8.0 it returns large_string
    assert reingest.schema().names == awards_players.schema().names

    reingest = reingest.order_by(["yearID", "playerID", "awardID", "lgID"])
    awards_players = awards_players.order_by(["yearID", "playerID", "awardID", "lgID"])

    backend.assert_frame_equal(reingest.to_pandas(), awards_players.to_pandas())


@pytest.mark.parametrize("ftype", ["csv", "parquet"])
def test_memtable_to_file(tmp_path, con, ftype, monkeypatch):
    """
    Tests against a regression spotted in #6091 where a `memtable` that is
    created and then immediately exported to `parquet` (or csv) will error
    because we weren't registering the in-memory table before trying to export
    it.
    """
    outfile = tmp_path / f"memtable.{ftype}"
    assert not outfile.is_file()

    monkeypatch.setattr(ibis.options, "default_backend", con)

    memtable = ibis.memtable({"col": [1, 2, 3, 4]})

    getattr(con, f"to_{ftype}")(memtable, outfile)

    assert outfile.is_file()


def test_table_to_csv(tmp_path, backend, awards_players):
    outcsv = tmp_path / "out.csv"

    # avoid pandas NaNonense
    awards_players = awards_players.select("playerID", "awardID", "yearID", "lgID")

    awards_players.to_csv(outcsv)

    df = pd.read_csv(outcsv, dtype=awards_players.schema().to_pandas())

    backend.assert_frame_equal(awards_players.to_pandas(), df)


@pytest.mark.parametrize(
    ("dtype", "pyarrow_dtype"),
    [
        param(
            dt.Decimal(38, 9),
            pa.Decimal128Type,
            id="decimal128",
            marks=[pytest.mark.notyet(["druid"], raises=sa.exc.ProgrammingError)],
        ),
        param(
            dt.Decimal(76, 38),
            pa.Decimal256Type,
            id="decimal256",
            marks=[
                pytest.mark.notyet(["impala"], reason="precision not supported"),
                pytest.mark.notyet(["duckdb"], reason="precision is out of range"),
                pytest.mark.notyet(
                    ["druid", "snowflake", "trino"], raises=sa.exc.ProgrammingError
                ),
                pytest.mark.notyet(["oracle"], raises=sa.exc.DatabaseError),
                pytest.mark.notyet(["mssql", "mysql"], raises=sa.exc.OperationalError),
                pytest.mark.notyet(
                    ["pyspark"],
                    raises=AnalysisException,
                    reason="precision is out of range",
                ),
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
        "impala",
        "mysql",
        "oracle",
        "postgres",
        "snowflake",
        "sqlite",
        "bigquery",
        "dask",
        "trino",
    ],
    raises=NotImplementedError,
    reason="read_delta not yet implemented",
)
@pytest.mark.notyet(["clickhouse"], raises=Exception)
@pytest.mark.notyet(["mssql", "pandas"], raises=PyDeltaTableError)
@pytest.mark.notyet(
    ["druid"],
    raises=pa.lib.ArrowTypeError,
    reason="arrow type conversion fails in `to_delta` call",
)
def test_roundtrip_delta(con, alltypes, tmp_path, monkeypatch):
    if con.name == "pyspark":
        pytest.importorskip("delta")
    else:
        pytest.importorskip("deltalake")

    t = alltypes.head()
    expected = t.to_pandas()
    path = tmp_path / "test.delta"
    t.to_delta(path)

    monkeypatch.setattr(ibis.options, "default_backend", con)
    dt = ibis.read_delta(path)
    result = dt.to_pandas()

    tm.assert_frame_equal(result, expected)


@pytest.mark.xfail_version(
    duckdb=["duckdb<0.8.1"], raises=AssertionError, reason="bug in duckdb"
)
@pytest.mark.notimpl(
    ["druid"], raises=AttributeError, reason="string type is used for timestamp_col"
)
@pytest.mark.notimpl(["mssql"], raises=pa.ArrowTypeError)
def test_arrow_timestamp_with_time_zone(alltypes):
    t = alltypes.select(
        tz=alltypes.timestamp_col.cast(
            alltypes.timestamp_col.type().copy(timezone="UTC")
        ),
        no_tz=alltypes.timestamp_col,
    ).limit(1)

    patype = PyArrowType.from_ibis(alltypes.timestamp_col.type())
    paunit = patype.unit
    expected = [pa.timestamp(paunit, tz="UTC"), pa.timestamp(paunit)]
    assert t.to_pyarrow().schema.types == expected

    with t.to_pyarrow_batches() as reader:
        (batch,) = reader
    assert batch.schema.types == expected


@pytest.mark.notimpl(["druid"])
@pytest.mark.notimpl(
    ["impala"], raises=AttributeError, reason="missing `fetchmany` on the cursor"
)
def test_to_torch(alltypes):
    import ibis.selectors as s

    torch = pytest.importorskip("torch")
    selector = s.numeric() | s.of_type("bool")
    numeric = alltypes.select(selector).limit(1)

    results = numeric.to_torch()

    assert all(isinstance(results[column], torch.Tensor) for column in numeric.columns)

    # torch can't handle non-numeric types
    non_numeric = alltypes.select(~selector).limit(1)
    with pytest.raises(TypeError):
        non_numeric.to_torch()


def test_empty_memtable(backend, con):
    expected = pd.DataFrame({"a": []})
    table = ibis.memtable(expected)
    result = con.execute(table)
    backend.assert_frame_equal(result, expected)


@pytest.mark.notimpl(["flink"])
def test_to_pandas_batches_empty_table(backend, con):
    t = backend.functional_alltypes.limit(0)
    n = t.count().execute()

    assert sum(map(len, con.to_pandas_batches(t))) == n
    assert sum(map(len, t.to_pandas_batches())) == n


@pytest.mark.notimpl(["druid", "flink"])
@pytest.mark.parametrize("n", [None, 1])
def test_to_pandas_batches_nonempty_table(backend, con, n):
    t = backend.functional_alltypes.limit(n)
    n = t.count().execute()

    assert sum(map(len, con.to_pandas_batches(t))) == n
    assert sum(map(len, t.to_pandas_batches())) == n


@pytest.mark.notimpl(["flink"])
@pytest.mark.parametrize("n", [None, 0, 1, 2])
def test_to_pandas_batches_column(backend, con, n):
    t = backend.functional_alltypes.limit(n).timestamp_col
    n = t.count().execute()

    assert sum(map(len, con.to_pandas_batches(t))) == n
    assert sum(map(len, t.to_pandas_batches())) == n


@pytest.mark.notimpl(["druid", "flink"])
def test_to_pandas_batches_scalar(backend, con):
    t = backend.functional_alltypes.timestamp_col.max()
    expected = t.execute()

    result1 = list(con.to_pandas_batches(t))
    assert result1 == [expected]

    result2 = list(t.to_pandas_batches())
    assert result2 == [expected]
