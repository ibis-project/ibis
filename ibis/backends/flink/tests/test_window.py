from __future__ import annotations

import pytest
from pytest import param

import ibis
from ibis.common.exceptions import UnsupportedOperationError


def test_window_requires_order_by(con, simple_table):
    expr = simple_table.mutate(simple_table.c - simple_table.c.mean())
    with pytest.raises(
        UnsupportedOperationError,
        match="Flink engine does not support generic window clause with no order by",
    ):
        con.compile(expr)


def test_window_does_not_support_multiple_order_by(con, simple_table):
    expr = simple_table.f.sum().over(
        rows=(-1, 1),
        group_by=[simple_table.g, simple_table.a],
        order_by=[simple_table.f, simple_table.d],
    )
    with pytest.raises(
        UnsupportedOperationError,
        match="Windows in Flink can only be ordered by a single time column",
    ):
        con.compile(expr)


def test_window_does_not_support_desc_order(con, simple_table):
    expr = simple_table.f.sum().over(
        rows=(-1, 1),
        group_by=[simple_table.g, simple_table.a],
        order_by=[simple_table.f.desc()],
    )
    with pytest.raises(
        UnsupportedOperationError,
        match="Flink only supports windows ordered in ASCENDING mode",
    ):
        con.compile(expr)


@pytest.mark.parametrize(
    ("window", "err"),
    [
        param(
            {"rows": (-1, 1)},
            "OVER RANGE FOLLOWING windows are not supported in Flink yet",
            id="bounded_rows_following",
        ),
        param(
            {"rows": (-1, None)},
            "OVER RANGE FOLLOWING windows are not supported in Flink yet",
            id="unbounded_rows_following",
        ),
        param(
            {"rows": (-500, 1)},
            "OVER RANGE FOLLOWING windows are not supported in Flink yet",
            id="casted_bounded_rows_following",
        ),
        param(
            {"range": (-1000, 0)},
            "Data Type mismatch between ORDER BY and RANGE clause",
            id="int_range",
        ),
    ],
)
def test_window_invalid_start_end(con, simple_table, window, err):
    expr = simple_table.f.sum().over(**window, order_by=simple_table.f)
    with pytest.raises(UnsupportedOperationError, match=err):
        con.compile(expr)


def test_range_window(con, snapshot, simple_table):
    expr = simple_table.f.sum().over(
        range=(-ibis.interval(minutes=500), 0), order_by=simple_table.f
    )
    result = con.compile(expr)
    snapshot.assert_match(result, "out.sql")


def test_rows_window(con, snapshot, simple_table):
    expr = simple_table.f.sum().over(rows=(-1000, 0), order_by=simple_table.f)
    result = con.compile(expr)
    snapshot.assert_match(result, "out.sql")
