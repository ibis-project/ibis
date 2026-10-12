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
from ibis.backends.tests.errors import (
    ExaQueryError,
    ImpalaHiveServer2Error,
    MySQLOperationalError,
    PsycoPg2InternalError,
    Py4JJavaError,
    PyDruidProgrammingError,
    PyODBCProgrammingError,
    SingleStoreDBOperationalError,
)

pytest.importorskip("pandas")

pytestmark = pytest.mark.notimpl(
    ["polars"],
    raises=com.OperationNotDefinedError,
    reason="window functions and scalar subqueries aren't implemented",
)

correlated_subquery_marks = [
    pytest.mark.notyet(
        ["mysql"],
        raises=MySQLOperationalError,
        reason="the subquery can't reference the outer query from a derived table",
    ),
    pytest.mark.notyet(
        ["exasol"],
        raises=ExaQueryError,
        reason="the subquery can't reference the outer query from a derived table",
    ),
    pytest.mark.notyet(
        ["impala"],
        raises=ImpalaHiveServer2Error,
        reason="the subquery can't reference the outer query from a derived table",
    ),
    pytest.mark.notyet(
        ["druid"],
        raises=PyDruidProgrammingError,
        reason="correlated subqueries aren't supported",
    ),
    pytest.mark.notyet(
        ["flink"], reason="unexpected correlate variable in the query plan"
    ),
]

row_number_frame_marks = [
    pytest.mark.notyet(
        ["risingwave"],
        raises=PsycoPg2InternalError,
        reason="ROW_NUMBER doesn't allow a frame clause",
    ),
    pytest.mark.notyet(
        ["druid"],
        raises=PyDruidProgrammingError,
        reason="ROW_NUMBER doesn't allow a frame clause",
    ),
]

t1 = ibis.memtable({"x": [1, 2, 3], "y": [10, 20, 30]})
t2 = ibis.memtable({"x": [2, 3, 4], "w": [1, 2, 3]})
t3 = ibis.memtable({"x": [2, 3], "z": [100, 200]})
t4 = ibis.memtable({"x": [3]})


def rows(con, expr):
    """Execute `expr` and return its rows sorted, as tuples."""
    df = con.execute(expr)
    df = df.astype(object).where(df.notna(), None)
    return sorted(df.itertuples(index=False, name=None))


@pytest.mark.notyet(
    ["mssql"],
    raises=PyODBCProgrammingError,
    reason="ROW_NUMBER requires an ORDER BY",
)
@pytest.mark.notyet(
    ["oracle"],
    raises=com.UnsupportedOperationError,
    reason="ROW_NUMBER requires an ORDER BY",
)
@pytest.mark.notyet(["flink"], raises=Py4JJavaError)
@pytest.mark.notyet(
    ["impala"],
    raises=AssertionError,
    reason="the rows are numbered before the join is applied",
)
@pytest.mark.notyet(
    ["risingwave"],
    raises=PsycoPg2InternalError,
    reason="ROW_NUMBER doesn't allow a frame clause",
)
@pytest.mark.notyet(
    ["druid"],
    raises=PyDruidProgrammingError,
    reason="ROW_NUMBER doesn't allow a frame clause",
)
def test_join_select_row_number(con):
    expr = t1.join(t2, "x").select("x", rn=ibis.row_number())
    # the order is unspecified, so only the numbers are checked
    assert sorted(rn for _, rn in rows(con, expr)) == [0, 1]


@pytest.mark.parametrize(
    ("make_expr", "expected"),
    [
        param(
            lambda: t1.join(t2, "x").select(
                "x", rn=ibis.row_number().over(order_by=t1.x)
            ),
            [(2, 0), (3, 1)],
            id="row_number_explicit_over",
            marks=row_number_frame_marks,
        ),
        param(
            lambda: t1.join(t2, "x").select("x", total=t1.y.sum()),
            [(2, 50), (3, 50)],
            id="reduction_with_column",
        ),
        param(
            lambda: t1.join(t2, "x").select(total=t1.y.sum()),
            [(50,), (50,)],
            id="reduction_only",
        ),
        param(
            lambda: t1.join(t2, "x").select("x", s=(t1.y * t2.w).sum()),
            [(2, 80), (3, 80)],
            id="reduction_over_both_sides",
        ),
        param(
            lambda: t1.join(t2, "x").select("x", s=t1.y.sum().over()),
            [(2, 50), (3, 50)],
            id="reduction_explicit_over",
        ),
        param(
            lambda: t1.join(t2, "x").select("x", s=t3.z.sum()),
            [(2, 300), (3, 300)],
            id="uncorrelated_subquery",
        ),
        param(
            lambda: t1.join(t2, "x").select("x", s=t3.filter(t3.x == t1.x).z.sum()),
            [(2, 100), (3, 200)],
            id="correlated_subquery",
            marks=correlated_subquery_marks,
        ),
        param(
            lambda: t1.semi_join(t2, "x").select("x", s=t2.w.sum()),
            # the semi-joined table is not part of the output, so a reduction
            # over it is a scalar subquery over the whole table
            [(2, 6), (3, 6)],
            id="semi_join_reduction_over_right",
            marks=[
                pytest.mark.notyet(
                    ["druid"],
                    raises=PyDruidProgrammingError,
                    reason="EXISTS isn't supported",
                ),
                pytest.mark.notyet(
                    ["flink"],
                    raises=Py4JJavaError,
                    reason="semi joins aren't supported",
                ),
            ],
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
            [(2, 20, 1, 50), (3, 30, 2, 50)],
            id="reduction",
        ),
        param(
            lambda: t1.join(t2, "x").mutate(s=t3.filter(t3.x == t1.x).z.sum()),
            [(2, 20, 1, 100), (3, 30, 2, 200)],
            id="correlated_subquery",
            marks=correlated_subquery_marks,
        ),
    ],
)
def test_join_mutate(con, make_expr, expected):
    expr = make_expr()
    assert rows(con, expr) == expected


impala_window_then_join = pytest.mark.notyet(
    ["impala"],
    raises=AssertionError,
    reason="the window function is evaluated after the join",
)


@pytest.mark.parametrize(
    ("make_expr", "expected"),
    [
        param(
            lambda: t1.join(t2, "x").select("x", s=t1.y.sum()),
            [(3, 50)],
            id="select_reduction",
            marks=impala_window_then_join,
        ),
        param(
            lambda: t1.join(t2, "x").select("x", s=t1.y.sum().over()),
            [(3, 50)],
            id="select_reduction_explicit_over",
            marks=impala_window_then_join,
        ),
        param(
            lambda: t1.join(t2, "x").select(
                "x", rn=ibis.row_number().over(order_by=t1.x)
            ),
            [(3, 1)],
            id="select_row_number_explicit_over",
            marks=row_number_frame_marks,
        ),
        param(
            lambda: t1.join(t2, "x").mutate(s=t1.y.sum()).select("x", "s"),
            [(3, 50)],
            id="mutate_reduction",
            marks=impala_window_then_join,
        ),
    ],
)
def test_window_then_join(con, make_expr, expected):
    expr = make_expr().join(t4, "x")
    assert rows(con, expr) == expected


@pytest.mark.parametrize(
    ("make_expr", "expected"),
    [
        param(
            lambda: t1.join(t2, "x").filter(t1.y < t3.filter(t3.x == t1.x).z.sum()),
            [(2, 20, 1), (3, 30, 2)],
            id="join_filter",
            marks=[
                *correlated_subquery_marks,
                pytest.mark.notyet(
                    ["singlestoredb"],
                    raises=SingleStoreDBOperationalError,
                    reason=(
                        "the subquery can't reference the outer query "
                        "from a derived table"
                    ),
                ),
                pytest.mark.notyet(
                    ["datafusion"],
                    raises=Exception,
                    reason="the physical optimizer fails with a schema mismatch",
                ),
            ],
        ),
        param(
            lambda: t1.filter(t1.y > 0).select("x", s=t3.filter(t3.x == t1.x).z.sum()),
            [(1, None), (2, 100), (3, 200)],
            id="filter_select",
            marks=correlated_subquery_marks,
        ),
    ],
)
def test_correlated_subquery(con, make_expr, expected):
    expr = make_expr()
    assert rows(con, expr) == expected
