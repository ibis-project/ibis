from __future__ import annotations

import pytest
from pytest import mark

import ibis
import ibis.common.exceptions as com
from ibis.conftest import IS_SPARK_REMOTE

pa = pytest.importorskip("pyarrow")
ds = pytest.importorskip("pyarrow.dataset")

pytestmark = [
    mark.notyet(
        ["databricks"],
        reason="Databricks does not support temporary tables, even though they allow the syntax",
    )
]


@mark.notimpl(["datafusion", "flink", "impala", "trino", "druid"])
@mark.notimpl(["exasol"], reason="Exasol does not support temporary tables")
@pytest.mark.never(
    ["risingwave"],
    raises=com.UnsupportedOperationError,
    reason="Feature is not yet implemented: CREATE TEMPORARY TABLE",
)
def test_persist_expression(backend, alltypes):
    non_persisted_table = alltypes.mutate(
        test_column=ibis.literal("calculation"), other_calc=ibis.literal("xyz")
    )
    persisted_table = non_persisted_table.cache()
    backend.assert_frame_equal(
        non_persisted_table.order_by("id").to_pandas(),
        persisted_table.order_by("id").to_pandas(),
    )


@mark.notimpl(["datafusion", "flink", "impala", "trino", "druid"])
@mark.notimpl(["exasol"], reason="Exasol does not support temporary tables")
@pytest.mark.never(
    ["risingwave"],
    raises=com.UnsupportedOperationError,
    reason="Feature is not yet implemented: CREATE TEMPORARY TABLE",
)
def test_persist_expression_contextmanager(backend, con, alltypes):
    non_cached_table = alltypes.mutate(
        test_column=ibis.literal("calculation"), other_column=ibis.literal("big calc")
    )
    with non_cached_table.cache() as cached_table:
        backend.assert_frame_equal(
            non_cached_table.order_by("id").to_pandas(),
            cached_table.order_by("id").to_pandas(),
        )
    assert non_cached_table.op() not in con._cache_op_to_entry


@mark.notimpl(["flink", "impala", "trino", "druid"])
@pytest.mark.never(
    ["risingwave"],
    raises=com.UnsupportedOperationError,
    reason="Feature is not yet implemented: CREATE TEMPORARY TABLE",
)
@mark.notimpl(["exasol"], reason="Exasol does not support temporary tables")
@mark.notyet(
    ["pyspark"],
    condition=not IS_SPARK_REMOTE,
    raises=AssertionError,
    reason=(
        "PySpark holds on to `cached_table` in the stack frame of an internal function. "
        "Caching works, but this test is probably too strict for PySpark. "
        "On the other hand, it's hard to write a test that makes assertions after weakref "
        "finalizers are called."
    ),
)
def test_persist_expression_multiple_refs(backend, con, alltypes):
    non_cached_table = alltypes.mutate(
        test_column=ibis.literal("calculation"), other_column=ibis.literal("big calc 2")
    )
    op = non_cached_table.op()
    cached_table = non_cached_table.cache()

    backend.assert_frame_equal(
        non_cached_table.order_by("id").to_pandas(),
        cached_table.order_by("id").to_pandas(),
        check_dtype=False,
    )

    name = cached_table.op().name
    nested_cached_table = non_cached_table.cache()

    # cached tables are identical and reusing the same op
    assert cached_table.op() is nested_cached_table.op()
    # table is cached
    assert op in con._cache_op_to_entry

    # deleting the first reference, leaves table in cache
    del nested_cached_table
    assert op in con._cache_op_to_entry

    # deleting the last reference, releases table from cache
    del cached_table
    assert op not in con._cache_op_to_entry

    # assert that table has been dropped
    assert name not in con.list_tables()


@mark.notimpl(["flink", "impala", "trino", "druid"])
@mark.notimpl(["exasol"], reason="Exasol does not support temporary tables")
@pytest.mark.never(
    ["risingwave"],
    raises=com.UnsupportedOperationError,
    reason="Feature is not yet implemented: CREATE TEMPORARY TABLE",
)
def test_persist_expression_repeated_cache(alltypes, con):
    non_cached_table = alltypes.mutate(
        test_column=ibis.literal("calculation"), other_column=ibis.literal("big calc 2")
    )
    cached_table = non_cached_table.cache()
    nested_cached_table = cached_table.cache()
    name = cached_table.op().name

    assert not nested_cached_table.to_pandas().empty

    del nested_cached_table, cached_table

    assert name not in con.list_tables()


@mark.notimpl(["flink", "impala", "trino", "druid"])
@mark.notimpl(["exasol"], reason="Exasol does not support temporary tables")
@pytest.mark.never(
    ["risingwave"],
    raises=com.UnsupportedOperationError,
    reason="Feature is not yet implemented: CREATE TEMPORARY TABLE",
)
def test_persist_expression_release(con, alltypes):
    non_cached_table = alltypes.mutate(
        test_column=ibis.literal("calculation"), other_column=ibis.literal("big calc 3")
    )
    cached_table = non_cached_table.cache()
    cached_table.release()

    assert non_cached_table.op() not in con._cache_op_to_entry

    # a second release does not hurt
    cached_table.release()

    with pytest.raises(Exception, match=cached_table.op().name):
        cached_table.execute()
