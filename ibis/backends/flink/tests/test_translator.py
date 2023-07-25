from __future__ import annotations

import pytest
from pytest import param

from ibis.backends.flink.compiler.core import translate


def test_translate_sum(snapshot, simple_table):
    expr = simple_table.a.sum()
    result = translate(expr.as_table().op())
    snapshot.assert_match(str(result), "out.sql")


def test_translate_count_star(snapshot, simple_table):
    expr = simple_table.group_by(simple_table.i).size()
    result = translate(expr.as_table().op())
    snapshot.assert_match(str(result), "out.sql")


@pytest.mark.parametrize(
    "unit",
    [
        param("ms", id="timestamp_ms"),
        param("s", id="timestamp_s"),
    ],
)
def test_translate_timestamp_from_unix(snapshot, simple_table, unit):
    expr = simple_table.d.to_timestamp(unit=unit)
    result = translate(expr.as_table().op())
    snapshot.assert_match(result, "out.sql")


def test_translate_complex_projections(snapshot, simple_table):
    expr = (
        simple_table.group_by(["a", "c"])
        .aggregate(the_sum=simple_table.b.sum())
        .group_by("a")
        .aggregate(mad=lambda x: x.the_sum.abs().mean())
    )
    result = translate(expr.as_table().op())
    snapshot.assert_match(result, "out.sql")


def test_translate_filter(snapshot, simple_table):
    expr = simple_table[
        ((simple_table.c > 0) | (simple_table.c < 0)) & simple_table.g.isin(["A", "B"])
    ]
    result = translate(expr.as_table().op())
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
def test_translate_extract_fields(snapshot, simple_table, kind):
    expr = getattr(simple_table.i, kind)().name("tmp")
    result = translate(expr.as_table().op())
    snapshot.assert_match(result, "out.sql")


def test_translate_complex_groupby_aggregation(snapshot, simple_table):
    keys = [simple_table.i.year().name("year"), simple_table.i.month().name("month")]
    b_unique = simple_table.b.nunique()
    expr = simple_table.group_by(keys).aggregate(
        total=simple_table.count(), b_unique=b_unique
    )
    result = translate(expr.as_table().op())
    snapshot.assert_match(result, "out.sql")


def test_translate_simple_filtered_agg(snapshot, simple_table):
    expr = simple_table.b.nunique(where=simple_table.g == "A")
    result = translate(expr.as_table().op())
    snapshot.assert_match(result, "out.sql")


def test_translate_complex_filtered_agg(snapshot, simple_table):
    expr = simple_table.group_by("b").aggregate(
        total=simple_table.count(),
        avg_a=simple_table.a.mean(),
        avg_a_A=simple_table.a.mean(where=simple_table.g == "A"),
        avg_a_B=simple_table.a.mean(where=simple_table.g == "B"),
    )
    result = translate(expr.as_table().op())
    snapshot.assert_match(result, "out.sql")


def test_translate_value_counts(snapshot, simple_table):
    expr = simple_table.i.year().value_counts()
    result = translate(expr.as_table().op())
    snapshot.assert_match(result, "out.sql")


def test_translate_having(snapshot, simple_table):
    expr = (
        simple_table.group_by("g")
        .having(simple_table.count() >= 1000)
        .aggregate(simple_table.b.sum().name("b_sum"))
    )
    result = translate(expr.as_table().op())
    snapshot.assert_match(result, "out.sql")
