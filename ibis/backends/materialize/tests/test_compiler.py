"""SQL compiler snapshot tests for Materialize backend.

These tests verify that Materialize-specific SQL compilation produces correct output.
"""

from __future__ import annotations

import pytest

import ibis
from ibis.backends.materialize import operations as mz_ops


@pytest.fixture
def simple_table():
    """Create a simple test table for snapshot tests."""
    return ibis.table(
        {
            "id": "int64",
            "value": "float64",
            "category": "string",
            "timestamp_col": "timestamp",
            "date_col": "date",
            "bool_col": "bool",
            "array_col": "array<int64>",
        },
        name="test_table",
    )


def test_mz_now_function(simple_table, assert_sql):
    """Test that mz_now() compiles to CAST(mz_now() AS TIMESTAMPTZ).

    Materialize's mz_now() returns mz_timestamp type, which we cast to
    TIMESTAMPTZ for compatibility with standard timestamp operations.
    """
    expr = simple_table.mutate(logical_time=mz_ops.MzNow())
    assert_sql(expr)


def test_mz_now_in_filter(simple_table, assert_sql):
    """Test mz_now() in temporal filter (idiomatic pattern).

    Best practice: Isolate mz_now() on one side of the comparison for
    optimal incremental computation.
    """
    expr = simple_table.filter(
        mz_ops.MzNow() > simple_table.timestamp_col + ibis.interval(days=1)
    )
    assert_sql(expr)


def test_distinct_on_rewrite_single_column(simple_table, assert_sql):
    """Test First() aggregate rewrites to DISTINCT ON (single column).

    When ALL aggregates are First(), Materialize compiler rewrites to
    DISTINCT ON which is more efficient than window functions.
    """
    expr = simple_table.group_by("category").aggregate(
        first_value=simple_table.value.first()
    )
    assert_sql(expr)


def test_distinct_on_rewrite_multiple_columns(simple_table, assert_sql):
    """Test First() aggregates rewrite to DISTINCT ON (multiple columns).

    Multiple First() aggregates should all be rewritten to DISTINCT ON
    selecting the appropriate columns.
    """
    expr = simple_table.group_by("category").aggregate(
        first_value=simple_table.value.first(),
        first_id=simple_table.id.first(),
    )
    assert_sql(expr)


def test_distinct_on_with_multiple_group_keys(simple_table, assert_sql):
    """Test DISTINCT ON rewrite with multiple grouping keys."""
    expr = simple_table.group_by(["category", "bool_col"]).aggregate(
        first_value=simple_table.value.first()
    )
    assert_sql(expr)


def test_array_length_instead_of_cardinality(simple_table, assert_sql):
    """Test array operations use array_length instead of cardinality.

    Materialize doesn't support cardinality(), so we use array_length(arr, 1).
    """
    expr = simple_table.mutate(array_len=simple_table.array_col.length())
    assert_sql(expr)


def test_array_index_with_array_length(simple_table, assert_sql):
    """Test array indexing uses array_length for bounds checking."""
    expr = simple_table.mutate(first_elem=simple_table.array_col[0])
    assert_sql(expr)


def test_array_slice_with_array_length(simple_table, assert_sql):
    """Test array slicing uses array_length for bounds."""
    expr = simple_table.mutate(sliced=simple_table.array_col[1:3])
    assert_sql(expr)


def test_array_repeat_with_order_by(simple_table, assert_sql):
    """Test array repeat uses ORDER BY for deterministic results.

    Materialize's generate_series returns unordered results, so we add ORDER BY.
    """
    expr = simple_table.mutate(repeated=simple_table.array_col.repeat(3))
    assert_sql(expr)


def test_sign_emulation_with_case_when(simple_table, assert_sql):
    """Test sign() function emulated with CASE WHEN.

    Materialize doesn't have sign(), so we use:
    CASE WHEN x = 0 THEN 0 WHEN x > 0 THEN 1 ELSE -1 END
    """
    expr = simple_table.mutate(value_sign=simple_table.value.sign())
    assert_sql(expr)


def test_range_without_sign_function(assert_sql):
    """Test range operation doesn't use sign() function.

    Materialize doesn't have sign(), so range operations must use
    CASE WHEN for determining direction.
    """
    expr = ibis.range(0, 10, 2)
    assert_sql(expr)


def test_interval_without_make_interval(simple_table, assert_sql):
    """Test interval creation without make_interval().

    Materialize doesn't support make_interval(), so we use
    CAST(value::text || ' days' AS INTERVAL) pattern.
    """
    expr = simple_table.mutate(
        future_date=simple_table.date_col + ibis.interval(days=7)
    )
    assert_sql(expr)


def test_date_from_ymd_without_make_date(assert_sql):
    """Test date construction without make_date().

    Materialize doesn't have make_date(), so we construct a date string
    and cast it: CAST('YYYY-MM-DD' AS DATE).
    """
    expr = ibis.date(2024, 1, 15)
    assert_sql(expr)


def test_date_literal_cast(assert_sql):
    """Test date literals are cast from ISO format strings."""
    from datetime import date

    expr = ibis.literal(date(2024, 1, 15))
    assert_sql(expr)


def test_timestamp_from_unix_seconds(simple_table, assert_sql):
    """Test timestamp from Unix seconds uses to_timestamp().

    Uses as_timestamp('s') method to convert Unix timestamps.
    """
    expr = simple_table.mutate(ts=simple_table.id.as_timestamp("s"))
    assert_sql(expr)


def test_timestamp_from_unix_milliseconds(simple_table, assert_sql):
    """Test timestamp from Unix milliseconds converts to seconds.

    Uses as_timestamp('ms') method which divides by 1000 before calling to_timestamp().
    """
    expr = simple_table.mutate(ts=simple_table.id.as_timestamp("ms"))
    assert_sql(expr)


def test_json_extract_with_cast_to_string(assert_sql):
    """Test JSON extraction can be converted to text using cast.

    Materialize doesn't have json_extract_path_text, but we can use
    bracket notation and cast to string.
    """
    t = ibis.table({"json_col": "json"}, name="json_table")
    expr = t.mutate(extracted=t.json_col["field1"]["field2"].cast("string"))
    assert_sql(expr)


def test_jsonb_typeof_function(assert_sql):
    """Test JSON type checking uses jsonb_typeof.

    Materialize only has jsonb_typeof, not json_typeof.
    """
    t = ibis.table({"json_col": "json"}, name="json_table")
    expr = t.mutate(json_type=t.json_col.typeof())
    assert_sql(expr)


def test_aggregate_without_first(simple_table, assert_sql):
    """Test normal aggregates work without First() rewriting."""
    expr = simple_table.group_by("category").aggregate(
        total=simple_table.value.sum(),
        count=simple_table.count(),
    )
    assert_sql(expr)


def test_window_function_row_number(simple_table, assert_sql):
    """Test ROW_NUMBER window function (workaround for Top-K).

    Since Materialize doesn't support some window functions, ROW_NUMBER
    is the recommended pattern for Top-K queries.
    """
    expr = simple_table.mutate(
        rank=ibis.row_number().over(
            ibis.window(group_by="category", order_by=ibis.desc("value"))
        )
    ).filter(lambda t: t.rank <= 3)
    assert_sql(expr)


def test_array_distinct(simple_table, assert_sql):
    """Test array distinct using ARRAY(SELECT DISTINCT UNNEST(array))."""
    expr = simple_table.mutate(unique_vals=simple_table.array_col.unique())
    assert_sql(expr)


def test_array_union(assert_sql):
    """Test array union concatenates and removes duplicates."""
    t = ibis.table({"arr1": "array<int64>", "arr2": "array<int64>"}, name="arrays")
    expr = t.mutate(combined=t.arr1.union(t.arr2))
    assert_sql(expr)


def test_cast_to_date_from_string(simple_table, assert_sql):
    """Test casting string to date."""
    expr = simple_table.mutate(parsed_date=simple_table.category.cast("date"))
    assert_sql(expr)


def test_date_now_operation(simple_table, assert_sql):
    """Test current_date operation.

    Materialize doesn't support CURRENT_DATE directly, uses NOW()::date.
    """
    expr = simple_table.mutate(today=ibis.now().date())
    assert_sql(expr)


def test_interval_from_integer_days(simple_table, assert_sql):
    """Test creating interval from integer (days).

    Uses as_interval('D') method to convert integers to day intervals.
    """
    expr = simple_table.mutate(
        future=simple_table.date_col + simple_table.id.as_interval("D")
    )
    assert_sql(expr)


def test_interval_from_integer_hours(simple_table, assert_sql):
    """Test creating interval from integer (hours).

    Uses as_interval('h') method to convert integers to hour intervals.
    """
    expr = simple_table.mutate(
        future=simple_table.timestamp_col + simple_table.id.as_interval("h")
    )
    assert_sql(expr)


def test_complex_aggregation_with_filter(simple_table, assert_sql):
    """Test aggregation with WHERE clause."""
    expr = (
        simple_table.filter(simple_table.value > 0)
        .group_by("category")
        .aggregate(total=simple_table.value.sum(), avg=simple_table.value.mean())
    )
    assert_sql(expr)


def test_subquery_with_distinct_on(simple_table, assert_sql):
    """Test subquery containing DISTINCT ON."""
    subquery = simple_table.group_by("category").aggregate(
        first_val=simple_table.value.first()
    )
    expr = subquery.filter(subquery.first_val > 100)
    assert_sql(expr)
