"""Validate the Feldera ad-hoc SQL surface against a live pipeline.

These tests are the source of truth for what our compiler may emit, because
``Backend.execute()`` / ``to_pyarrow()`` / ``raw_sql()`` all run ad-hoc SQL
via ``Pipeline.query_arrow``, and ad-hoc SQL is parsed by **Apache
DataFusion** — not Apache Calcite (see
``../feldera-all/feldera/docs.feldera.com/docs/sql/ad-hoc.md``).

Run::

    docker compose up feldera -d
    uv run pytest ibis/backends/feldera/tests/test_adhoc_surface.py -p no:cacheprovider -v
"""

from __future__ import annotations

import os
import time
import uuid
from contextlib import contextmanager

import pandas as pd
import pyarrow as pa
import pytest

HOST = os.environ.get("FELDERA_HOST", os.environ.get("IBIS_TEST_FELDERA_HOST", "http://localhost:8080"))

pytestmark = pytest.mark.feldera

_BASE_SQL = """\
CREATE TABLE t (
    a INTEGER NOT NULL,
    b DOUBLE,
    c VARCHAR,
    ts TIMESTAMP
) WITH ('materialized' = 'true');
"""

_ROW = {"a": 3, "b": 1.5, "c": "a,b,c", "ts": pd.Timestamp("2024-01-15 03:04:05")}


@contextmanager
def _pipeline(sql_program: str = _BASE_SQL, row: dict | None = _ROW):
    """Create a pipeline, push one row into ``t``, yield it, then tear down."""
    from feldera import FelderaClient, PipelineBuilder

    client = FelderaClient(HOST)
    name = f"ibis-surface-{uuid.uuid4().hex[:8]}"
    pipe = PipelineBuilder(client, name=name, sql=sql_program).create(wait=True)
    pipe.start()
    if row is not None:
        pipe.input_pandas("t", pd.DataFrame([row]))
        # Give the pipeline a moment to ingest before querying.
        time.sleep(1.5)
    try:
        yield pipe
    finally:
        try:
            pipe.stop(force=True)
        except Exception:
            pass
        for _ in range(10):
            try:
                pipe.delete()
                break
            except Exception:
                time.sleep(1)


def _query(pipe, sql: str) -> pa.Table:
    batches = list(pipe.query_arrow(sql))
    if not batches:
        return pa.table({})
    return pa.Table.from_batches(batches, batches[0].schema)


def _first(table: pa.Table):
    """Return the first value of the first column."""
    cols = table.to_pydict()
    assert cols, "query returned no columns"
    first_col = next(iter(cols.values()))
    assert first_col, "query returned no rows"
    return first_col[0]


# ----- String split -----------------------------------------------------------


def test_string_to_array_accepted():
    """Postgres's ``string_to_array`` is what our compiler emits for
    ``StringSplit`` (via the inherited ``sge.Split → string_to_array``
    transform).  DataFusion ad-hoc accepts it."""
    with _pipeline() as pipe:
        out = _query(pipe, "SELECT string_to_array(c, ',') AS p FROM t")
    assert _first(out) == ["a", "b", "c"]


def test_split_function_rejected():
    """``SPLIT(string, delimiter)`` is documented for the Calcite program
    dialect but is **not** available in ad-hoc (DataFusion) queries."""
    with _pipeline() as pipe:
        with pytest.raises(Exception, match="Invalid function 'split'"):
            _query(pipe, "SELECT SPLIT(c, ',') AS p FROM t")


# ----- Intervals --------------------------------------------------------------


def test_interval_literal_inline_unit():
    """``INTERVAL '3 DAY'`` — the form emitted for ``ibis.interval(days=3)``."""
    with _pipeline() as pipe:
        out = _query(pipe, "SELECT ts + INTERVAL '3 DAY' AS d FROM t")
    assert _first(out).isoformat() == "2024-01-18T03:04:05"


def test_interval_literal_bare_unit():
    """``INTERVAL '3' DAY`` — SQL-standard style (also accepted)."""
    with _pipeline() as pipe:
        out = _query(pipe, "SELECT ts + INTERVAL '3' DAY AS d FROM t")
    assert _first(out).isoformat() == "2024-01-18T03:04:05"


def test_interval_cast_string():
    """The ``IntervalFromInteger`` lowering:
    ``CAST(CAST(a AS VARCHAR) || ' days' AS INTERVAL)``."""
    with _pipeline() as pipe:
        out = _query(
            pipe,
            "SELECT ts + CAST(CAST(a AS VARCHAR) || ' days' AS INTERVAL) AS d FROM t",
        )
    assert _first(out).isoformat() == "2024-01-18T03:04:05"


# ----- Date / timestamp construction ------------------------------------------


def test_make_date_accepted():
    """``MAKE_DATE`` — emitted by the inherited ``visit_DateFromYMD``."""
    with _pipeline() as pipe:
        out = _query(pipe, "SELECT MAKE_DATE(2024, 1, 2) AS d FROM t")
    assert str(_first(out)) == "2024-01-02"


def test_make_timestamp_rejected():
    """``MAKE_TIMESTAMP`` is rejected by ad-hoc (DataFusion) — this is why
    ``visit_TimestampFromYMDHMS`` overrides the Postgres default."""
    with _pipeline() as pipe:
        with pytest.raises(Exception, match="Invalid function 'make_timestamp'"):
            _query(pipe, "SELECT MAKE_TIMESTAMP(2024, 1, 2, 3, 4, 5) AS ts2 FROM t")


def test_to_timestamp_accepted():
    """``TO_TIMESTAMP`` — the function our ``visit_TimestampFromYMDHMS``
    emits."""
    with _pipeline() as pipe:
        out = _query(pipe, "SELECT TO_TIMESTAMP('2024-01-02 03:04:05') AS ts2 FROM t")
    assert _first(out).isoformat() == "2024-01-02T03:04:05"


def test_cast_string_as_date():
    """``CAST('...' AS DATE)`` — the form the base literal lowering would
    emit via ``datefromparts`` → ``MAKE_DATE``; both are accepted."""
    with _pipeline() as pipe:
        out = _query(pipe, "SELECT CAST('2024-01-15' AS DATE) AS d FROM t")
    assert str(_first(out)) == "2024-01-15"


# ----- Type emission: REAL vs DOUBLE -----------------------------------------


def test_real_type_accepted():
    """``REAL`` (single-precision float) — what ``FLOAT`` maps to in our
    dialect."""
    from feldera import FelderaClient, PipelineBuilder

    client = FelderaClient(HOST)
    name = f"ibis-types-{uuid.uuid4().hex[:8]}"
    pipe = PipelineBuilder(
        client,
        name=name,
        sql="CREATE TABLE x (f REAL, d DOUBLE PRECISION) WITH ('materialized' = 'true');",
    ).create(wait=True)
    try:
        cols = {f["name"]: f["columntype"]["type"] for f in pipe.tables()[0].fields}
        assert cols["f"] == "REAL", cols
        assert cols["d"] == "DOUBLE", cols
    finally:
        try:
            pipe.stop(force=True)
        except Exception:
            pass
        for _ in range(10):
            try:
                pipe.delete()
                break
            except Exception:
                time.sleep(1)
