from __future__ import annotations

import pytest
from pytest import param

import ibis


def test_sum(con, snapshot, simple_table):
    expr = simple_table.a.sum()
    result = con.compile(expr)
    snapshot.assert_match(str(result), "out.sql")


def test_count_star(con, snapshot, simple_table):
    expr = simple_table.group_by(simple_table.i).size()
    result = con.compile(expr)
    snapshot.assert_match(str(result), "out.sql")


@pytest.mark.parametrize(
    "unit",
    [
        param("ms", id="timestamp_ms"),
        param("s", id="timestamp_s"),
    ],
)
def test_timestamp_from_unix(con, snapshot, simple_table, unit):
    expr = simple_table.d.to_timestamp(unit=unit)
    result = con.compile(expr)
    snapshot.assert_match(result, "out.sql")


def test_complex_projections(con, snapshot, simple_table):
    expr = (
        simple_table.group_by(["a", "c"])
        .aggregate(the_sum=simple_table.b.sum())
        .group_by("a")
        .aggregate(mad=lambda x: x.the_sum.abs().mean())
    )
    result = con.compile(expr)
    snapshot.assert_match(result, "out.sql")


def test_filter(con, snapshot, simple_table):
    expr = simple_table[
        ((simple_table.c > 0) | (simple_table.c < 0)) & simple_table.g.isin(["A", "B"])
    ]
    result = con.compile(expr)
    snapshot.assert_match(result, "out.sql")


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
def test_extract_fields(con, snapshot, simple_table, kind):
    expr = getattr(simple_table.i, kind)().name("tmp")
    result = con.compile(expr)
    snapshot.assert_match(result, "out.sql")


def test_complex_groupby_aggregation(con, snapshot, simple_table):
    keys = [simple_table.i.year().name("year"), simple_table.i.month().name("month")]
    b_unique = simple_table.b.nunique()
    expr = simple_table.group_by(keys).aggregate(
        total=simple_table.count(), b_unique=b_unique
    )
    result = con.compile(expr)
    snapshot.assert_match(result, "out.sql")


def test_simple_filtered_agg(con, snapshot, simple_table):
    expr = simple_table.b.nunique(where=simple_table.g == "A")
    result = con.compile(expr)
    snapshot.assert_match(result, "out.sql")


def test_complex_filtered_agg(con, snapshot, simple_table):
    expr = simple_table.group_by("b").aggregate(
        total=simple_table.count(),
        avg_a=simple_table.a.mean(),
        avg_a_A=simple_table.a.mean(where=simple_table.g == "A"),
        avg_a_B=simple_table.a.mean(where=simple_table.g == "B"),
    )
    result = con.compile(expr)
    snapshot.assert_match(result, "out.sql")


def test_value_counts(con, snapshot, simple_table):
    expr = simple_table.i.year().value_counts()
    result = con.compile(expr)
    snapshot.assert_match(result, "out.sql")


def test_having(con, snapshot, simple_table):
    expr = (
        simple_table.group_by("g")
        .having(simple_table.count() >= 1000)
        .aggregate(simple_table.b.sum().name("b_sum"))
    )
    result = con.compile(expr)
    snapshot.assert_match(result, "out.sql")


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
def test_tvf(con, snapshot, simple_table, function_type, params):
    expr = getattr(simple_table.window_by(time_col=simple_table.i), function_type)(
        **params
    )
    result = con.compile(expr)
    snapshot.assert_match(result, "out.sql")
