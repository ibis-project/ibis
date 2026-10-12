"""Window functions and reductions in projections of joins.

`Join.select` follows the same rules as `Table.select`:

- analytic functions and reductions over the joined tables become window
  functions over the join result, so the output has one row per joined row
- reductions over tables outside the join become scalar subqueries, which
  are correlated when they reference a joined table

A window function is evaluated over the join it was selected from, even if
more tables are joined afterwards.
"""

from __future__ import annotations

import pytest
from pytest import param

import ibis
import ibis.common.exceptions as com

pytest.importorskip("pyarrow")

pytestmark = pytest.mark.notimpl(
    ["polars"],
    raises=com.OperationNotDefinedError,
    reason="window functions and scalar subqueries aren't implemented",
)


t1 = ibis.memtable({"x": [1, 2, 3], "y": [10.0, 20.0, 30.0]})
t2 = ibis.memtable({"x": [2, 3, 4], "w": [1.0, 2.0, 3.0]})
t3 = ibis.memtable({"x": [2, 3], "z": [100.0, 200.0]})
t4 = ibis.memtable({"x": [3]})


def rows(con, expr):
    """Execute `expr` and return its rows sorted, as tuples."""
    return sorted(tuple(row.values()) for row in con.to_pyarrow(expr).to_pylist())


@pytest.mark.parametrize(
    ("make_expr", "expected"),
    [
        param(
            lambda: t1.join(t2, "x").select("x", rn=ibis.row_number()),
            # the order is unspecified, but every row gets a distinct number
            [(2, 0), (3, 1)],
            id="row_number",
        ),
        param(
            lambda: t1.join(t2, "x").select(
                "x", rn=ibis.row_number().over(order_by=t1.x)
            ),
            [(2, 0), (3, 1)],
            id="row_number_explicit_over",
        ),
        param(
            lambda: t1.join(t2, "x").select("x", total=t1.y.sum()),
            [(2, 50.0), (3, 50.0)],
            id="reduction_with_column",
        ),
        param(
            lambda: t1.join(t2, "x").select(total=t1.y.sum()),
            [(50.0,), (50.0,)],
            id="reduction_only",
        ),
        param(
            lambda: t1.join(t2, "x").select("x", s=(t1.y * t2.w).sum()),
            [(2, 80.0), (3, 80.0)],
            id="reduction_over_both_sides",
        ),
        param(
            lambda: t1.join(t2, "x").select("x", s=t1.y.sum().over()),
            [(2, 50.0), (3, 50.0)],
            id="reduction_explicit_over",
        ),
        param(
            lambda: t1.join(t2, "x").select("x", s=t3.z.sum()),
            [(2, 300.0), (3, 300.0)],
            id="uncorrelated_subquery",
        ),
        param(
            lambda: t1.join(t2, "x").select("x", s=t3.filter(t3.x == t1.x).z.sum()),
            [(2, 100.0), (3, 200.0)],
            id="correlated_subquery",
        ),
        param(
            lambda: t1.semi_join(t2, "x").select("x", s=t2.w.sum()),
            # the semi-joined table is not part of the output, so a reduction
            # over it is a scalar subquery over the whole table
            [(2, 6.0), (3, 6.0)],
            id="semi_join_reduction_over_right",
        ),
    ],
)
def test_join_select(con, make_expr, expected):
    expr = make_expr()
    assert rows(con, expr) == expected


@pytest.mark.parametrize(
    ("make_expr", "expected"),
    [
        param(
            lambda: t1.join(t2, "x").mutate(total=t1.y.sum()),
            [(2, 20.0, 1.0, 50.0), (3, 30.0, 2.0, 50.0)],
            id="reduction",
        ),
        param(
            lambda: t1.join(t2, "x").mutate(s=t3.filter(t3.x == t1.x).z.sum()),
            [(2, 20.0, 1.0, 100.0), (3, 30.0, 2.0, 200.0)],
            id="correlated_subquery",
        ),
    ],
)
def test_join_mutate(con, make_expr, expected):
    expr = make_expr()
    assert rows(con, expr) == expected


@pytest.mark.parametrize(
    ("make_expr", "expected"),
    [
        param(
            lambda: t1.join(t2, "x").select("x", s=t1.y.sum()),
            [(3, 50.0)],
            id="select_reduction",
        ),
        param(
            lambda: t1.join(t2, "x").select("x", s=t1.y.sum().over()),
            [(3, 50.0)],
            id="select_reduction_explicit_over",
        ),
        param(
            lambda: t1.join(t2, "x").select(
                "x", rn=ibis.row_number().over(order_by=t1.x)
            ),
            [(3, 1)],
            id="select_row_number_explicit_over",
        ),
        param(
            lambda: t1.join(t2, "x").mutate(s=t1.y.sum()).select("x", "s"),
            [(3, 50.0)],
            id="mutate_reduction",
        ),
    ],
)
def test_window_then_join(con, make_expr, expected):
    expr = make_expr().join(t4, "x")
    assert rows(con, expr) == expected


def test_join_filter_correlated_subquery(con):
    expr = t1.join(t2, "x").filter(t1.y < t3.filter(t3.x == t1.x).z.sum())
    assert rows(con, expr) == [(2, 20.0, 1.0), (3, 30.0, 2.0)]


def test_filter_select_correlated_subquery(con):
    expr = t1.filter(t1.y > 0).select("x", s=t3.filter(t3.x == t1.x).z.sum())
    assert rows(con, expr) == [(1, None), (2, 100.0), (3, 200.0)]
