from __future__ import annotations

import pytest
from pyflink.util.exceptions import TableException
from pytest import param

import ibis
from ibis import _
from ibis.backends.tests.errors import Py4JJavaError


@pytest.mark.xfail(raises=TableException)
def test_window_requires_order_by(con):
    t = con.tables.functional_alltypes
    expr = t.mutate(t.double_col - t.double_col.mean())
    con.execute(expr)


@pytest.mark.xfail(raises=TableException)
def test_window_does_not_support_multiple_order_by(con):
    t = con.tables.functional_alltypes
    expr = t.double_col.sum().over(rows=(-1, 1), order_by=[t.timestamp_col, t.int_col])
    con.execute(expr)


@pytest.mark.parametrize(
    "window",
    [
        param(
            {"rows": (-1, 1)},
            id="bounded_rows_following",
            marks=[pytest.mark.xfail(raises=TableException)],
        ),
        param(
            {"rows": (-1, None)},
            id="unbounded_rows_following",
            marks=[pytest.mark.xfail(raises=TableException)],
        ),
        param(
            {"rows": (-500, 1)},
            id="casted_bounded_rows_following",
            marks=[pytest.mark.xfail(raises=TableException)],
        ),
        param(
            {"range": (-1000, 0)},
            id="int_range",
            marks=[pytest.mark.xfail(raises=Py4JJavaError)],
        ),
    ],
)
def test_window_invalid_start_end(con, window):
    t = con.tables.functional_alltypes
    expr = t.int_col.sum().over(**window, order_by=t.timestamp_col)
    con.execute(expr)


def test_range_window(simple_table, assert_sql):
    expr = simple_table.f.sum().over(
        range=(-ibis.interval(minutes=500), 0), order_by=simple_table.f
    )
    assert_sql(expr)


def test_rows_window(simple_table, assert_sql):
    expr = simple_table.f.sum().over(rows=(-1000, 0), order_by=simple_table.f)
    assert_sql(expr)


def test_tumble_window_by_agg(con):
    t = con.table("payment_msg")
    expr = (
        t.window_by(time_col=t.createTime)
        .tumble(size=ibis.interval(seconds=30))
        .agg(by=["provinceId"], avgPayAmount=_.payAmount.mean())
    )
    result = expr.to_pandas()
    assert list(result.columns) == [
        "window_start",
        "window_end",
        "provinceId",
        "avgPayAmount",
    ]
