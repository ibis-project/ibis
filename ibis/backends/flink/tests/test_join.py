from __future__ import annotations

import tempfile

import pandas as pd
import pytest

import ibis
import ibis.expr.datatypes as dt
import ibis.expr.schema as sch
from ibis.backends.tests.errors import Py4JJavaError


@pytest.mark.broken(
    ["flink"],
    raises=Py4JJavaError,
    reason=(
        """
    py4j.protocol.Py4JJavaError: An error occurred while calling z:org.apache.flink.table.runtime.arrow.ArrowUtils.collectAsPandasDataFrame.
    : org.apache.flink.table.api.ValidationException: Temporal table join currently only supports 'FOR SYSTEM_TIME AS OF' left table's time attribute field

    This error is thrown when the watermark is not correctly set for the column
    used in the join predicate. In this case, left and right tables are created
    from in-memory. This is why, they are actually create as temporary views not
    tables. Hence the error raised while executing join.

    The goal with this test is to exercise the failure. This is why it does not
    assert.
    """
    ),
)
def test_temporal_join_w_in_memory_objs(con):
    schema = sch.Schema(
        {
            "key": dt.string,
            "value": dt.float32,
            "time": dt.timestamp(scale=9),
        }
    )

    pd_left = pd.DataFrame(
        {
            "key": ["x", "x", "x", "x"],
            "value": [1.1, 2.2, 3.3, 4.4],
            "time": pd.to_datetime([1, 2, 3, 4]),
        }
    )
    table_left = con.create_table(
        "table_left",
        pd_left,
        schema=schema,
        watermark=ibis.watermark(
            time_col="time", allowed_delay=ibis.interval(seconds=15)
        ),
        temp=True,
    )

    pd_right = pd.DataFrame(
        {"key": ["x", "x"], "value": [1.2, 2.0], "time": pd.to_datetime([2, 4])}
    )
    table_right = con.create_table(
        "table_right",
        pd_right,
        schema=schema,
        watermark=ibis.watermark(
            time_col="time", allowed_delay=ibis.interval(seconds=15)
        ),
        temp=True,
    )

    expr = table_left.asof_join(
        table_right,
        predicates=[
            table_left["key"] == table_right["key"],
            table_left["time"] >= table_right["time"],
        ],
    )
    con.compile(expr)
    expr.to_pandas()


def test_temporal_join(data_dir, con, tempdir_sink_configs):
    schema = sch.Schema(
        {
            "id": dt.int32,
            "bool_col": dt.bool,
            "smallint_col": dt.int16,
            "int_col": dt.int32,
            "timestamp_col": dt.timestamp(scale=3),
        }
    )

    path = f"{data_dir}/parquet/functional_alltypes.parquet"
    df = pd.read_parquet(path)
    df = df[list(schema.names)]
    df = df.head()

    # Create `table_left`
    table_left_tempdir = tempfile.TemporaryDirectory()
    table_left = con.create_table(
        "table_left",
        schema=schema,
        tbl_properties=tempdir_sink_configs(table_left_tempdir.name),
        watermark=ibis.watermark(
            time_col="timestamp_col", allowed_delay=ibis.interval(seconds=15)
        ),
    )
    con.insert(
        "table_left",
        obj=df,
        schema=schema,
    ).wait()

    # Create `table_left`
    table_right_tempdir = tempfile.TemporaryDirectory()
    table_right = con.create_table(
        "table_right",
        schema=schema,
        tbl_properties=tempdir_sink_configs(table_right_tempdir.name),
        watermark=ibis.watermark(
            time_col="timestamp_col", allowed_delay=ibis.interval(seconds=15)
        ),
        primary_key="id",
    )
    con.insert(
        "table_right",
        obj=df,
        schema=schema,
    ).wait()

    # Join `table_left` and `table_right`
    expr = table_left.asof_join(
        table_right,
        predicates=[
            table_left["id"] == table_right["id"],
            table_left["timestamp_col"] >= table_right["timestamp_col"],
        ],
    )
    sql = con.compile(expr)

    expected_sql = """SELECT t0.`id`, t0.`bool_col`, t0.`smallint_col`, t0.`int_col`,
       t0.`timestamp_col`, CAST(t1.`id` AS int) AS `id_right`,
       t1.`bool_col` AS `bool_col_right`,
       t1.`smallint_col` AS `smallint_col_right`,
       t1.`int_col` AS `int_col_right`,
       t1.`timestamp_col` AS `timestamp_col_right`
FROM table_left AS t0
  JOIN table_right FOR SYSTEM_TIME AS OF t0.`timestamp_col` AS t1
    ON (t0.`id` = t1.`id`)"""
    assert sql == expected_sql

    join_df = expr.to_pandas().sort_values("id")

    table_left_tempdir.cleanup()
    table_right_tempdir.cleanup()

    expected_df_to_str = (
        "  id  bool_col  smallint_col  int_col           timestamp_col  id_right  bool_col_right  smallint_col_right  int_col_right     timestamp_col_right\n"
        "6690      True             0        0 2010-11-01 00:00:00.000      6690            True                   0              0 2010-11-01 00:00:00.000\n"
        "6691     False             1        1 2010-11-01 00:01:00.000      6691           False                   1              1 2010-11-01 00:01:00.000\n"
        "6692      True             2        2 2010-11-01 00:02:00.100      6692            True                   2              2 2010-11-01 00:02:00.100\n"
        "6693     False             3        3 2010-11-01 00:03:00.300      6693           False                   3              3 2010-11-01 00:03:00.300\n"
        "6694      True             4        4 2010-11-01 00:04:00.600      6694            True                   4              4 2010-11-01 00:04:00.600"
    )
    assert join_df.to_string(index=False) == expected_df_to_str
