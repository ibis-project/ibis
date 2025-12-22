from __future__ import annotations

import pytest

import ibis
from ibis.backends.tests.errors import DuckDBInvalidInputException

pd = pytest.importorskip("pandas")
tm = pytest.importorskip("pandas.testing")


@pytest.fixture(scope="module")
def time_df1():
    return pd.DataFrame(
        {
            "time": pd.to_datetime([1, 2, 3, 4], unit="s"),
            "value": [1.1, 2.2, 3.3, 4.4],
            "group": ["a", "a", "a", "a"],
        }
    )


@pytest.fixture(scope="module")
def time_df2():
    return pd.DataFrame(
        {
            "time": pd.to_datetime([2, 4], unit="s"),
            "other_value": [1.2, 2.0],
            "group": ["a", "a"],
        }
    )


@pytest.fixture(scope="module")
def time_keyed_df1():
    return pd.DataFrame(
        {
            "time": pd.Series(
                pd.date_range(start="2017-01-02 01:02:03.234", periods=6)
            ),
            "key": [1, 2, 3, 1, 2, 3],
            "value": [1.2, 1.4, 2.0, 4.0, 8.0, 16.0],
        }
    )


@pytest.fixture(scope="module")
def time_keyed_df2():
    return pd.DataFrame(
        {
            "time": pd.Series(
                pd.date_range(start="2017-01-02 01:02:03.234", freq="3D", periods=3)
            ),
            "key": [1, 2, 3],
            "other_value": [1.1, 1.2, 2.2],
        }
    )


@pytest.fixture(scope="module")
def time_left(time_df1):
    return ibis.memtable(time_df1)


@pytest.fixture(scope="module")
def time_right(time_df2):
    return ibis.memtable(time_df2)


@pytest.fixture(scope="module")
def time_keyed_left(time_keyed_df1):
    return ibis.memtable(time_keyed_df1)


@pytest.fixture(scope="module")
def time_keyed_right(time_keyed_df2):
    return ibis.memtable(time_keyed_df2)


mark_direction_on = pytest.mark.parametrize(
    ("direction", "on_func"),
    [
        pytest.param("backward", lambda left, right: left >= right, id="ge"),
        pytest.param("forward", lambda left, right: left <= right, id="le"),
        pytest.param("backward", lambda _left, _right: "time", id="string"),
    ],
)


@mark_direction_on
@pytest.mark.notyet(
    "clickhouse",
    reason="does not support asof joins that do not also have an equality predicate on the join.",
)
@pytest.mark.notyet(
    [
        "datafusion",
        "trino",
        "mysql",
        "pyspark",
        "druid",
        "impala",
        "bigquery",
        "exasol",
        "oracle",
        "mssql",
        "sqlite",
        "flink",
        "databricks",
        "athena",
    ]
)
def test_asof_join(con, time_left, time_right, time_df1, time_df2, direction, on_func):
    on = on_func(time_left["time"], time_right["time"])
    expr = time_left.asof_join(time_right, on)

    result = con.execute(expr)
    expected = pd.merge_asof(
        time_df1.drop(columns=["group"]),
        time_df2.drop(columns=["group"]),
        on="time",
        direction=direction,
    )

    result = result.sort_values(["time"]).reset_index(drop=True)
    expected = expected.sort_values(["time"]).reset_index(drop=True)

    # duckdb returns datetime64[us], pandas defaults to use datetime64[ns]
    tm.assert_frame_equal(result[expected.columns], expected, check_dtype=False)
    with pytest.raises(AssertionError):
        tm.assert_series_equal(result["time"], result["time_right"])


@mark_direction_on
@pytest.mark.notyet(
    [
        "datafusion",
        "trino",
        "mysql",
        "pyspark",
        "druid",
        "impala",
        "bigquery",
        "exasol",
        "oracle",
        "mssql",
        "sqlite",
        "flink",
        "databricks",
        "athena",
    ]
)
def test_noop_keyed_asof_join(
    con, time_left, time_right, time_df1, time_df2, direction, on_func
):
    on = on_func(time_left["time"], time_right["time"])
    expr = time_left.asof_join(time_right, on, "group")

    result = con.execute(expr)
    expected = pd.merge_asof(
        time_df1, time_df2, on="time", by="group", direction=direction
    )

    result = result.sort_values(["time"]).reset_index(drop=True)
    expected = expected.sort_values(["time"]).reset_index(drop=True)

    # duckdb returns datetime64[us], pandas defaults to use datetime64[ns]
    tm.assert_frame_equal(result[expected.columns], expected, check_dtype=False)
    with pytest.raises(AssertionError):
        tm.assert_series_equal(result["time"], result["time_right"])


@mark_direction_on
@pytest.mark.notimpl(
    ["clickhouse"], raises=AssertionError, reason="`time` is truncated to seconds"
)
@pytest.mark.notyet(
    [
        "datafusion",
        "trino",
        "mysql",
        "pyspark",
        "druid",
        "impala",
        "bigquery",
        "exasol",
        "oracle",
        "mssql",
        "sqlite",
        "flink",
        "databricks",
        "athena",
    ]
)
def test_keyed_asof_join(
    con,
    time_keyed_left,
    time_keyed_right,
    time_keyed_df1,
    time_keyed_df2,
    direction,
    on_func,
):
    on = on_func(time_keyed_left["time"], time_keyed_right["time"])
    expr = time_keyed_left.asof_join(time_keyed_right, on, "key")

    result = con.execute(expr)
    expected = pd.merge_asof(
        time_keyed_df1, time_keyed_df2, on="time", by="key", direction=direction
    )

    result = result.sort_values(["key", "time"]).reset_index(drop=True)
    expected = expected.sort_values(["key", "time"]).reset_index(drop=True)

    # duckdb returns datetime64[us], pandas defaults to use datetime64[ns]
    tm.assert_frame_equal(result[expected.columns], expected, check_dtype=False)
    with pytest.raises(AssertionError):
        tm.assert_series_equal(result["time"], result["time_right"])


@mark_direction_on
@pytest.mark.parametrize("right_column_key", [None, "on_right_time"])
@pytest.mark.notimpl(
    ["clickhouse"], raises=AssertionError, reason="`time` is truncated to seconds"
)
@pytest.mark.notyet(
    [
        "bigquery",
        "databricks",
        "datafusion",
        "druid",
        "exasol",
        "flink",
        "impala",
        "mssql",
        "mysql",
        "oracle",
        "pyspark",
        "sqlite",
        "trino",
        "athena",
    ]
)
@pytest.mark.xfail_version(
    duckdb=["duckdb>=0.10.2,<1.1.1"], raises=DuckDBInvalidInputException
)
def test_keyed_asof_join_with_tolerance(
    con,
    time_keyed_left,
    time_keyed_right,
    time_keyed_df1,
    time_keyed_df2,
    direction,
    on_func,
    right_column_key,
):
    left_key = right_key = "time"
    if right_column_key:
        right_key = right_column_key
        time_keyed_right = time_keyed_right.rename({right_column_key: "time"})
        assert left_key != right_key

    on = on_func(time_keyed_left[left_key], time_keyed_right[right_key])
    if (right_column_key is not None) and isinstance(on, str) and on == "time":
        pytest.xfail(
            "The right column was renamed away from the original name of `time`."
        )
    expr = time_keyed_left.asof_join(
        time_keyed_right, on, "key", tolerance=ibis.interval(days=2)
    )

    result = con.execute(expr)
    expected = pd.merge_asof(
        time_keyed_df1,
        time_keyed_df2,
        on="time",
        by="key",
        tolerance=pd.Timedelta("2D"),
        direction=direction,
    )

    result = result.sort_values(["key", "time"]).reset_index(drop=True)
    expected = expected.sort_values(["key", "time"]).reset_index(drop=True)

    tm.assert_frame_equal(
        # drop `time` from comparison to avoid issues with different time resolution
        result[expected.columns].drop(["time"], axis=1),
        expected.drop(["time"], axis=1),
    )

    # check that time is equal in value, if not dtype
    tm.assert_series_equal(result["time"], expected["time"], check_dtype=False)
