from __future__ import annotations

from operator import methodcaller

import pytest
from packaging.version import parse as vparse
from pytest import param

import ibis
import ibis.expr.datatypes as dt
from ibis import util
from ibis.backends.tests.errors import (
    DatabricksServerOperationError,
    DuckDBNotImplementedException,
    DuckDBParserException,
    ExaQueryError,
    MySQLOperationalError,
    OracleDatabaseError,
    PyDeltaTableError,
    PyDruidProgrammingError,
    PyODBCProgrammingError,
    PySparkArithmeticException,
    PySparkParseException,
    SnowflakeProgrammingError,
    TrinoUserError,
)
from ibis.conftest import CI, IS_SPARK_REMOTE

pd = pytest.importorskip("pandas")
pa = pytest.importorskip("pyarrow")

limit = [param(42, id="limit")]

no_limit = [param(None, id="nolimit")]

limit_no_limit = limit + no_limit


@pytest.mark.skipif(
    vparse(pa.__version__) < vparse("14"), reason="pyarrow >= 14 required"
)
def test_table___arrow_c_stream__(awards_players):
    sol = awards_players.to_pyarrow()
    res = pa.table(awards_players)
    assert res.schema.equals(sol.schema)
    assert len(res) == len(sol)

    # With explicit schema
    schema = awards_players.schema().to_pyarrow()
    res = pa.table(awards_players, schema=schema)
    assert res.schema.equals(sol.schema)
    assert len(res) == len(sol)


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


@pytest.mark.notimpl(["datafusion", "flink"])
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


@pytest.mark.notimpl(["datafusion", "flink"])
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


@pytest.mark.notimpl(
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

    backend.assert_frame_equal(
        awards_players.to_pandas().fillna(pd.NA), df.fillna(pd.NA)
    )


def test_table_to_parquet_dir(tmp_path, backend, awards_players):
    outparquet_dir = tmp_path / "out"

    if backend.name() == "pyspark":
        if IS_SPARK_REMOTE:
            pytest.skip("writes to remote output directory")
        # pyspark already writes more than one file
        awards_players.to_parquet_dir(outparquet_dir)
    else:
        # max_ force pyarrow to write more than one parquet file
        awards_players.to_parquet_dir(
            outparquet_dir, max_rows_per_file=3000, max_rows_per_group=3000
        )

    parquet_files = sorted(
        outparquet_dir.glob("*.parquet"),
        key=lambda path: int(path.with_suffix("").name.split("-")[1]),
    )

    sort_keys = list(awards_players.columns)

    expected = (
        pd.concat(map(pd.read_parquet, parquet_files))
        .sort_values(sort_keys)
        .reset_index(drop=True)
    )
    result = awards_players.to_pandas().sort_values(sort_keys).reset_index(drop=True)
    backend.assert_frame_equal(result, expected)


@pytest.mark.notimpl(
    ["duckdb"],
    reason="cannot inline WriteOptions objects",
    raises=DuckDBNotImplementedException,
)
@pytest.mark.parametrize("version", ["1.0", "2.6"])
def test_table_to_parquet_writer_kwargs(version, tmp_path, backend, awards_players):
    outparquet = tmp_path / "out.parquet"
    awards_players.to_parquet(outparquet, version=version)

    df = pd.read_parquet(outparquet)

    backend.assert_frame_equal(
        awards_players.to_pandas().fillna(pd.NA), df.fillna(pd.NA)
    )

    md = pa.parquet.read_metadata(outparquet)

    assert md.format_version == version


@pytest.mark.notimpl(
    [
        "bigquery",
        "clickhouse",
        "datafusion",
        "impala",
        "mssql",
        "mysql",
        "oracle",
        "polars",
        "postgres",
        "risingwave",
        "pyspark",
        "snowflake",
        "sqlite",
        "trino",
        "databricks",
    ],
    reason="no partitioning support",
)
@pytest.mark.notimpl(["druid", "flink"], reason="No to_parquet support")
@pytest.mark.notimpl(["exasol"], raises=TypeError)
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
    assert reingest.schema().keys() == awards_players.schema().keys()

    reingest = reingest.order_by(["yearID", "playerID", "awardID", "lgID"])
    awards_players = awards_players.order_by(["yearID", "playerID", "awardID", "lgID"])

    # reorder columns to match the partitioning
    backend.assert_frame_equal(
        reingest.to_pandas(), awards_players[reingest.columns].to_pandas()
    )


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


@pytest.mark.notimpl(
    ["duckdb"],
    reason="cannot inline WriteOptions objects",
    raises=DuckDBParserException,
)
@pytest.mark.parametrize("delimiter", [";", "\t"], ids=["semicolon", "tab"])
def test_table_to_csv_writer_kwargs(delimiter, tmp_path, awards_players):
    import pyarrow.csv as pcsv

    outcsv = tmp_path / "out.csv"
    # avoid pandas NaNonense
    awards_players = awards_players.select("playerID", "awardID", "yearID", "lgID")

    awards_players.to_csv(outcsv, write_options=pcsv.WriteOptions(delimiter=delimiter))
    df = pd.read_csv(outcsv, delimiter=delimiter, nrows=1)
    assert len(df) == 1


@pytest.mark.parametrize(
    ("dtype", "pyarrow_dtype"),
    [
        param(
            dt.Decimal(38, 9),
            pa.Decimal128Type,
            id="decimal128",
            marks=[pytest.mark.notyet(["exasol"], raises=ExaQueryError)],
        ),
        param(
            dt.Decimal(76, 38),
            pa.Decimal256Type,
            id="decimal256",
            marks=[
                pytest.mark.notyet(["impala"], reason="precision not supported"),
                pytest.mark.notyet(["duckdb"], reason="precision is out of range"),
                pytest.mark.notyet(["mssql"], raises=PyODBCProgrammingError),
                pytest.mark.notyet(["snowflake"], raises=SnowflakeProgrammingError),
                pytest.mark.notyet(["trino"], raises=TrinoUserError),
                pytest.mark.notyet(["oracle"], raises=OracleDatabaseError),
                pytest.mark.notyet(["mysql"], raises=MySQLOperationalError),
                pytest.mark.notyet(
                    ["pyspark"],
                    raises=(PySparkParseException, PySparkArithmeticException),
                    reason="precision is out of range",
                ),
                pytest.mark.notyet(["exasol"], raises=ExaQueryError),
                pytest.mark.notyet(
                    ["databricks"], raises=DatabricksServerOperationError
                ),
            ],
        ),
    ],
)
def test_to_pyarrow_decimal(backend, dtype, pyarrow_dtype):
    if backend.name() == "polars":
        pytest.skip("polars crashes the interpreter")

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
        "flink",
        "impala",
        "mysql",
        "oracle",
        "postgres",
        "risingwave",
        "snowflake",
        "sqlite",
        "bigquery",
        "trino",
        "exasol",
        "druid",
        "databricks",  # feels a bit weird given it's their format ¯\_(ツ)_/¯
    ],
    raises=NotImplementedError,
    reason="read_delta not yet implemented",
)
@pytest.mark.notyet(["clickhouse"], raises=Exception)
@pytest.mark.notyet(["mssql"], raises=PyDeltaTableError)
@pytest.mark.xfail_version(
    pyspark=["pyspark<4"],
    condition=CI and IS_SPARK_REMOTE,
    reason="not supported until pyspark 4",
)
def test_roundtrip_delta(backend, con, alltypes, tmp_path, monkeypatch):
    if con.name == "pyspark":
        pytest.importorskip("delta")
    else:
        pytest.importorskip("deltalake", exc_type=ImportError)

    t = alltypes.head()
    expected = t.to_pandas()
    path = tmp_path / "test.delta"
    t.to_delta(path)

    monkeypatch.setattr(ibis.options, "default_backend", con)
    dt = ibis.read_delta(path)
    result = dt.to_pandas()

    backend.assert_frame_equal(result, expected)


@pytest.mark.notimpl(
    ["druid"],
    raises=PyDruidProgrammingError,
    reason="Invalid SQL generated; druid doesn't know about TIMESTAMPTZ",
)
@pytest.mark.notimpl(
    ["databricks"], raises=AssertionError, reason="Only the devil knows"
)
def test_arrow_timestamp_with_time_zone(alltypes):
    from ibis.formats.pyarrow import PyArrowType

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


@pytest.mark.notimpl(["druid", "flink"])
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


@pytest.mark.notimpl(["flink"])
@pytest.mark.notyet(
    ["druid"],
    raises=PyDruidProgrammingError,
    reason="backend doesn't support an empty VALUES construct",
)
def test_empty_memtable(backend, con):
    expected = pd.DataFrame({"a": []})
    table = ibis.memtable(expected)
    result = con.execute(table)
    backend.assert_frame_equal(result, expected)


def test_to_pandas_batches_empty_table(backend, con):
    t = backend.functional_alltypes.limit(0)
    n = t.count().execute()

    assert sum(map(len, con.to_pandas_batches(t))) == n
    assert sum(map(len, t.to_pandas_batches())) == n


@pytest.mark.parametrize("n", [None, 1])
def test_to_pandas_batches_nonempty_table(backend, con, n):
    t = backend.functional_alltypes.limit(n)
    n = t.count().execute()

    assert sum(map(len, con.to_pandas_batches(t))) == n
    assert sum(map(len, t.to_pandas_batches())) == n


@pytest.mark.parametrize("n", [None, 0, 1, 2])
def test_to_pandas_batches_column(backend, con, n):
    t = backend.functional_alltypes.limit(n).timestamp_col
    n = t.count().execute()

    assert sum(map(len, con.to_pandas_batches(t))) == n
    assert sum(map(len, t.to_pandas_batches())) == n


def test_to_pandas_batches_scalar(backend, con):
    t = backend.functional_alltypes.int_col.max()
    expected = t.execute()

    result1 = list(con.to_pandas_batches(t))
    assert result1 == [expected]

    result2 = list(t.to_pandas_batches())
    assert result2 == [expected]


@pytest.mark.parametrize("limit", limit_no_limit)
@pytest.mark.never(
    ["druid"], raises=AssertionError, reason="Druid has an extra __time column"
)
def test_table_to_polars(limit, awards_players):
    pl = pytest.importorskip("polars")
    res = awards_players.to_polars(limit=limit)
    assert isinstance(res, pl.DataFrame)
    if limit is not None:
        assert len(res) == limit

    expected_schema = {
        "playerID": pl.Utf8,
        "awardID": pl.Utf8,
        "yearID": pl.Int64,
        "lgID": pl.Utf8,
        "tie": pl.Utf8,
        "notes": pl.Utf8,
    }
    assert res.schema == expected_schema


@pytest.mark.parametrize("limit", limit_no_limit)
@pytest.mark.parametrize(
    ("output_format", "expected_column_type"),
    [("pyarrow", "ChunkedArray"), ("polars", "Series")],
    ids=["pyarrow", "polars"],
)
def test_column_to_memory(limit, awards_players, output_format, expected_column_type):
    mod = pytest.importorskip(output_format)
    method = methodcaller(f"to_{output_format}", limit=limit)
    res = method(awards_players.awardID)
    assert isinstance(res, getattr(mod, expected_column_type))
    assert (
        (len(res) == limit)
        if limit is not None
        else len(res) == awards_players.count().execute()
    )


@pytest.mark.parametrize("limit", limit_no_limit)
def test_column_to_list(limit, awards_players):
    res = awards_players.awardID.to_list(limit=limit)
    assert isinstance(res, list)
    assert (
        (len(res) == limit)
        if limit is not None
        else len(res) == awards_players.count().execute()
    )


@pytest.mark.parametrize("limit", no_limit)
@pytest.mark.parametrize(
    ("output_format", "converter"),
    [("pyarrow", methodcaller("as_py")), ("polars", lambda x: x)],
    ids=["pyarrow", "polars"],
)
def test_scalar_to_memory(limit, awards_players, output_format, converter):
    pytest.importorskip(output_format)
    method = methodcaller(f"to_{output_format}", limit=limit)
    scalar = method(awards_players.yearID.min())
    assert isinstance(converter(scalar), int)

    expr = awards_players.filter(awards_players.awardID == "DEADBEEF").yearID.min()
    res = method(expr)
    assert converter(res) is None
