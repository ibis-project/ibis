from __future__ import annotations

import pandas as pd
import pandas.testing as tm
import pytest

import ibis
from ibis import _
from ibis.expr.rewrites import lower_pivot_wider


@pytest.fixture(scope="module")
def fish(con):
    data = pd.DataFrame(
        {
            "fish": [1, 1, 2, 2, 3],
            "station": ["Release", "Lisbon", "Release", "Release", "Lisbon"],
            "site": ["a", "a", "b", "a", "b"],
            "seen": [1, 1, 1, 1, 1],
            "n": [3, 4, 5, 6, 7],
        }
    )
    return con.create_table("pivot_fish", data, temp=True, overwrite=True)


def execute_generic(con, expr):
    """Execute `expr` through the filtered-aggregate lowering instead of PIVOT."""
    return con.execute(expr.op().replace(lower_pivot_wider).to_expr())


def assert_pivot_equal(result, expected):
    # normalize None/NaN so missing pivot cells compare equal regardless of dtype
    result = result.astype(object).where(result.notna(), None)
    expected = expected.astype(object).where(expected.notna(), None)
    tm.assert_frame_equal(result, expected, check_dtype=False)


def test_native_pivot(con, fish, snapshot):
    expr = fish.pivot_wider(
        id_cols=["fish", "seen"],
        names_from="station",
        values_from="n",
        values_agg="sum",
        names_sort=True,
    )
    sql = ibis.to_sql(expr)
    snapshot.assert_match(sql, "out.sql")
    assert "PIVOT(" in sql

    result = con.execute(expr).sort_values("fish").reset_index(drop=True)
    expected = pd.DataFrame(
        {
            "fish": [1, 2, 3],
            "seen": [1, 1, 1],
            "Lisbon": [4.0, None, 7.0],
            "Release": [3.0, 11.0, None],
        }
    )
    assert_pivot_equal(result, expected)
    assert_pivot_equal(result, execute_generic(con, expr))


def test_multiple_values_fill(con, fish, snapshot):
    expr = fish.pivot_wider(
        id_cols="fish",
        names_from="station",
        values_from=["seen", "n"],
        values_agg="sum",
        values_fill=0,
        names_sort=True,
    )
    snapshot.assert_match(ibis.to_sql(expr), "out.sql")
    assert expr.columns == (
        "fish",
        "seen_Lisbon",
        "n_Lisbon",
        "seen_Release",
        "n_Release",
    )

    result = con.execute(expr).sort_values("fish").reset_index(drop=True)
    expected = pd.DataFrame(
        {
            "fish": [1, 2, 3],
            "seen_Lisbon": [1, 0, 1],
            "n_Lisbon": [4, 0, 7],
            "seen_Release": [1, 2, 0],
            "n_Release": [3, 11, 0],
        }
    )
    assert_pivot_equal(result, expected)
    assert_pivot_equal(result, execute_generic(con, expr))


def test_explicit_names_and_prefix(con, fish):
    expr = fish.pivot_wider(
        id_cols="fish",
        names=["Release", "Nowhere"],
        names_from="station",
        values_from="n",
        values_agg="sum",
        names_prefix="at",
    )
    assert expr.columns == ("fish", "at_Release", "at_Nowhere")

    result = con.execute(expr).sort_values("fish").reset_index(drop=True)
    expected = pd.DataFrame(
        {"fish": [1, 2, 3], "at_Release": [3.0, 11.0, None], "at_Nowhere": [None] * 3}
    )
    assert_pivot_equal(result, expected)


def test_no_id_cols(con, fish, snapshot):
    expr = fish.pivot_wider(
        id_cols=[],
        names_from="station",
        values_from="n",
        values_agg=_.count(),
        names_sort=True,
    )
    snapshot.assert_match(ibis.to_sql(expr), "out.sql")
    result = con.execute(expr)
    expected = pd.DataFrame({"Lisbon": [2], "Release": [3]})
    assert_pivot_equal(result, expected)


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        pytest.param(
            dict(values_agg=lambda v: v.sum(where=v > 3)),
            "PIVOT ignores an aggregate's own FILTER",
            id="filtered_reduction",
        ),
        pytest.param(
            dict(names_from=["station", "site"]),
            "PIVOT builds the cross product of several FOR columns",
            id="multiple_names_from",
        ),
    ],
)
def test_falls_back_to_filtered_aggregates(con, fish, kwargs, reason):
    kwargs = dict(id_cols="fish", names_from="station", values_from="n") | kwargs
    expr = fish.pivot_wider(names_sort=True, **kwargs)
    sql = ibis.to_sql(expr)
    assert "PIVOT(" not in sql, reason
    assert "FILTER" in sql

    result = con.execute(expr).sort_values("fish").reset_index(drop=True)
    assert_pivot_equal(result, execute_generic(con, expr))
