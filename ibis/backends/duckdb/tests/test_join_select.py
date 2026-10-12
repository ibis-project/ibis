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

import duckdb
import pytest
from pytest import param

import ibis

pytest.importorskip("pyarrow")


@pytest.fixture(scope="module")
def con():
    return ibis.duckdb.connect()


@pytest.fixture(scope="module")
def t1(con):
    return con.create_table("t1", {"x": [1, 2, 3], "y": [10.0, 20.0, 30.0]})


@pytest.fixture(scope="module")
def t2(con):
    return con.create_table("t2", {"x": [2, 3, 4], "w": [1.0, 2.0, 3.0]})


@pytest.fixture(scope="module")
def t3(con):
    return con.create_table("t3", {"x": [2, 3], "z": [100.0, 200.0]})


@pytest.fixture(scope="module")
def t4(con):
    return con.create_table("t4", {"x": [3]})


def rows(expr):
    """Execute `expr` and return its rows sorted, as tuples."""
    return sorted(tuple(row.values()) for row in expr.to_pyarrow().to_pylist())


@pytest.mark.parametrize(
    ("make_expr", "expected"),
    [
        param(
            lambda t1, t2, _: t1.join(t2, "x").select("x", rn=ibis.row_number()),
            # the order is unspecified, but every row gets a distinct number
            [(2, 0), (3, 1)],
            id="row_number",
        ),
        param(
            lambda t1, t2, _: t1.join(t2, "x").select(
                "x", rn=ibis.row_number().over(order_by=t1.x)
            ),
            [(2, 0), (3, 1)],
            id="row_number_explicit_over",
        ),
        param(
            lambda t1, t2, _: t1.join(t2, "x").select("x", total=t1.y.sum()),
            [(2, 50.0), (3, 50.0)],
            id="reduction_with_column",
        ),
        param(
            lambda t1, t2, _: t1.join(t2, "x").select(total=t1.y.sum()),
            [(50.0,), (50.0,)],
            id="reduction_only",
        ),
        param(
            lambda t1, t2, _: t1.join(t2, "x").select("x", s=(t1.y * t2.w).sum()),
            [(2, 80.0), (3, 80.0)],
            id="reduction_over_both_sides",
        ),
        param(
            lambda t1, t2, _: t1.join(t2, "x").select("x", s=t1.y.sum().over()),
            [(2, 50.0), (3, 50.0)],
            id="reduction_explicit_over",
        ),
        param(
            lambda t1, t2, t3: t1.join(t2, "x").select("x", s=t3.z.sum()),
            [(2, 300.0), (3, 300.0)],
            id="uncorrelated_subquery",
        ),
        param(
            lambda t1, t2, t3: t1.join(t2, "x").select(
                "x", s=t3.filter(t3.x == t1.x).z.sum()
            ),
            [(2, 100.0), (3, 200.0)],
            id="correlated_subquery",
            marks=pytest.mark.xfail(
                raises=duckdb.BinderException,
                reason="the subquery references a table missing from FROM",
            ),
        ),
        param(
            lambda t1, t2, _: t1.semi_join(t2, "x").select("x", s=t2.w.sum()),
            # the semi-joined table is not part of the output, so a reduction
            # over it is a scalar subquery over the whole table
            [(2, 6.0), (3, 6.0)],
            id="semi_join_reduction_over_right",
        ),
    ],
)
def test_join_select(t1, t2, t3, make_expr, expected):
    expr = make_expr(t1, t2, t3)
    assert rows(expr) == expected


@pytest.mark.parametrize(
    ("make_expr", "expected"),
    [
        param(
            lambda t1, t2, _: t1.join(t2, "x").mutate(total=t1.y.sum()),
            [(2, 20.0, 1.0, 50.0), (3, 30.0, 2.0, 50.0)],
            id="reduction",
        ),
        param(
            lambda t1, t2, t3: t1.join(t2, "x").mutate(
                s=t3.filter(t3.x == t1.x).z.sum()
            ),
            [(2, 20.0, 1.0, 100.0), (3, 30.0, 2.0, 200.0)],
            id="correlated_subquery",
            marks=pytest.mark.xfail(
                raises=duckdb.BinderException,
                reason="the subquery references a table missing from FROM",
            ),
        ),
    ],
)
def test_join_mutate(t1, t2, t3, make_expr, expected):
    expr = make_expr(t1, t2, t3)
    assert rows(expr) == expected


@pytest.mark.parametrize(
    ("make_expr", "expected"),
    [
        param(
            lambda t1, t2: t1.join(t2, "x").select("x", s=t1.y.sum()),
            [(3, 50.0)],
            id="select_reduction",
            marks=pytest.mark.xfail(
                raises=AssertionError,
                reason="the window is evaluated over the later join",
            ),
        ),
        param(
            lambda t1, t2: t1.join(t2, "x").select("x", s=t1.y.sum().over()),
            [(3, 50.0)],
            id="select_reduction_explicit_over",
            marks=pytest.mark.xfail(
                raises=AssertionError,
                reason="the window is evaluated over the later join",
            ),
        ),
        param(
            lambda t1, t2: t1.join(t2, "x").select(
                "x", rn=ibis.row_number().over(order_by=t1.x)
            ),
            [(3, 1)],
            id="select_row_number_explicit_over",
            marks=pytest.mark.xfail(
                raises=AssertionError,
                reason="the window is evaluated over the later join",
            ),
        ),
        param(
            lambda t1, t2: t1.join(t2, "x").mutate(s=t1.y.sum()).select("x", "s"),
            [(3, 50.0)],
            id="mutate_reduction",
        ),
    ],
)
def test_window_then_join(t1, t2, t4, make_expr, expected):
    expr = make_expr(t1, t2).join(t4, "x")
    assert rows(expr) == expected


@pytest.mark.xfail(
    raises=duckdb.BinderException,
    reason="the subquery references a table missing from FROM",
)
def test_join_filter_correlated_subquery(t1, t2, t3):
    expr = t1.join(t2, "x").filter(t1.y < t3.filter(t3.x == t1.x).z.sum())
    assert rows(expr) == [(2, 20.0, 1.0), (3, 30.0, 2.0)]


def test_filter_select_correlated_subquery(t1, t3):
    expr = t1.filter(t1.y > 0).select("x", s=t3.filter(t3.x == t1.x).z.sum())
    assert rows(expr) == [(1, None), (2, 100.0), (3, 200.0)]
