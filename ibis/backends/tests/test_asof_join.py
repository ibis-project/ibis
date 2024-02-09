from __future__ import annotations

import operator

import pandas as pd
import pandas.testing as tm
import pytest

import ibis


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


@pytest.mark.parametrize(
    ("direction", "op"), [("backward", operator.ge), ("forward", operator.le)]
)
@pytest.mark.notyet(
    [
        "datafusion",
        "snowflake",
        "trino",
        "postgres",
        "mysql",
        "pyspark",
        "druid",
        "impala",
        "bigquery",
        "exasol",
        "oracle",
        "mssql",
        "sqlite",
        "risingwave",
        "flink",
    ]
)
def test_asof_join(con, time_left, time_right, time_df1, time_df2, direction, op):
    on = op(time_left["time"], time_right["time"])
    expr = time_left.asof_join(time_right, on=on, predicates="group")

    result = con.execute(expr)
    expected = pd.merge_asof(
        time_df1, time_df2, on="time", by="group", direction=direction
    )

    result = result.sort_values(["group", "time"]).reset_index(drop=True)
    expected = expected.sort_values(["group", "time"]).reset_index(drop=True)

    tm.assert_frame_equal(result[expected.columns], expected)
    with pytest.raises(AssertionError):
        tm.assert_series_equal(result["time"], result["time_right"])


@pytest.mark.parametrize(
    ("direction", "op"), [("backward", operator.ge), ("forward", operator.le)]
)
@pytest.mark.broken(
    ["clickhouse"], raises=AssertionError, reason="`time` is truncated to seconds"
)
@pytest.mark.notyet(
    [
        "datafusion",
        "snowflake",
        "trino",
        "postgres",
        "mysql",
        "pyspark",
        "druid",
        "impala",
        "bigquery",
        "exasol",
        "oracle",
        "mssql",
        "sqlite",
        "risingwave",
        "flink",
    ]
)
def test_keyed_asof_join_with_tolerance(
    con,
    time_keyed_left,
    time_keyed_right,
    time_keyed_df1,
    time_keyed_df2,
    direction,
    op,
):
    on = op(time_keyed_left["time"], time_keyed_right["time"])
    expr = time_keyed_left.asof_join(
        time_keyed_right, on=on, by="key", tolerance=ibis.interval(days=2)
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

    tm.assert_frame_equal(result[expected.columns], expected)
    with pytest.raises(AssertionError):
        tm.assert_series_equal(result["time"], result["time_right"])
    with pytest.raises(AssertionError):
        tm.assert_series_equal(result["key"], result["key_right"])
