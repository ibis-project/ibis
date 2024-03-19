from __future__ import annotations

import pytest
from pytest import param

import ibis
from ibis.common.deferred import _


def test_sum(con, simple_table, assert_sql):
    expr = simple_table.a.sum()
    assert_sql(expr)


def test_count_star(con, simple_table, assert_sql):
    expr = simple_table.group_by(simple_table.i).size()
    assert_sql(expr)


@pytest.mark.parametrize(
    "unit",
    [
        param("ms", id="timestamp_ms"),
        param("s", id="timestamp_s"),
    ],
)
def test_timestamp_from_unix(con, simple_table, unit, assert_sql):
    expr = simple_table.d.to_timestamp(unit=unit)
    assert_sql(expr)


def test_complex_projections(con, simple_table, assert_sql):
    expr = (
        simple_table.group_by(["a", "c"])
        .aggregate(the_sum=simple_table.b.sum())
        .group_by("a")
        .aggregate(mad=lambda x: x.the_sum.abs().mean())
    )
    assert_sql(expr)


def test_filter(con, simple_table, assert_sql):
    expr = simple_table[
        ((simple_table.c > 0) | (simple_table.c < 0)) & simple_table.g.isin(["A", "B"])
    ]
    assert_sql(expr)


@pytest.mark.parametrize(
    "kind",
    [
        "year",
        "quarter",
        "month",
        "week_of_year",
        "day_of_year",
        "day",
        "hour",
        "minute",
        "second",
    ],
)
def test_extract_fields(con, simple_table, kind, assert_sql):
    expr = getattr(simple_table.i, kind)().name("tmp")
    assert_sql(expr)


def test_complex_groupby_aggregation(con, simple_table, assert_sql):
    keys = [simple_table.i.year().name("year"), simple_table.i.month().name("month")]
    b_unique = simple_table.b.nunique()
    expr = simple_table.group_by(keys).aggregate(
        total=simple_table.count(), b_unique=b_unique
    )
    assert_sql(expr)


def test_simple_filtered_agg(con, simple_table, assert_sql):
    expr = simple_table.b.nunique(where=simple_table.g == "A")
    assert_sql(expr)


def test_complex_filtered_agg(con, snapshot, simple_table, assert_sql):
    expr = simple_table.group_by("b").aggregate(
        total=simple_table.count(),
        avg_a=simple_table.a.mean(),
        avg_a_A=simple_table.a.mean(where=simple_table.g == "A"),
        avg_a_B=simple_table.a.mean(where=simple_table.g == "B"),
    )
    assert_sql(expr)


def test_value_counts(con, simple_table, assert_sql):
    expr = simple_table.i.year().value_counts()
    assert_sql(expr)


def test_having(con, simple_table, assert_sql):
    expr = (
        simple_table.group_by("g")
        .having(simple_table.count() >= 1000)
        .aggregate(simple_table.b.sum().name("b_sum"))
    )
    assert_sql(expr)


@pytest.mark.parametrize(
    "function_type,params",
    [
        pytest.param(
            "tumble", {"window_size": ibis.interval(minutes=15)}, id="tumble_window"
        ),
        pytest.param(
            "hop",
            {
                "window_size": ibis.interval(minutes=15),
                "window_slide": ibis.interval(minutes=1),
            },
            id="hop_window",
        ),
        pytest.param(
            "cumulate",
            {
                "window_size": ibis.interval(minutes=1),
                "window_step": ibis.interval(seconds=10),
            },
            id="cumulate_window",
        ),
    ],
)
def test_windowing_tvf(con, simple_table, function_type, params, assert_sql):
    expr = getattr(simple_table.window_by(time_col=simple_table.i), function_type)(
        **params
    )
    assert_sql(expr)


def test_window_aggregation(con, simple_table, assert_sql):
    expr = (
        simple_table.window_by(time_col=simple_table.i)
        .tumble(window_size=ibis.interval(minutes=15))
        .group_by(["window_start", "window_end", "g"])
        .aggregate(mean=_.d.mean())
    )
    assert_sql(expr)


def test_window_topn(con, simple_table, assert_sql):
    expr = simple_table.window_by(time_col="i").tumble(
        window_size=ibis.interval(seconds=600),
    )["a", "b", "c", "d", "g", "window_start", "window_end"]
    expr = expr.mutate(
        rownum=ibis.row_number().over(
            group_by=["window_start", "window_end"], order_by=ibis.desc("g")
        )
    )
    expr = expr[expr.rownum <= 3]
    assert_sql(expr)
