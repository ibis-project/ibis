from __future__ import annotations

import random

import pandas as pd
import pytest
import sqlalchemy as sa
from pytest import param

import ibis
import ibis.common.exceptions as com
import ibis.expr.types as ir
from ibis import _


@pytest.fixture
def union_subsets(alltypes, df):
    cols_a, cols_b, cols_c = (alltypes.columns.copy() for _ in range(3))

    random.seed(89)
    random.shuffle(cols_a)
    random.shuffle(cols_b)
    random.shuffle(cols_c)
    assert cols_a != cols_b != cols_c

    a = alltypes.filter((_.id >= 5200) & (_.id <= 5210))[cols_a]
    b = alltypes.filter((_.id >= 5205) & (_.id <= 5215))[cols_b]
    c = alltypes.filter((_.id >= 5213) & (_.id <= 5220))[cols_c]

    da = df[(df.id >= 5200) & (df.id <= 5210)][cols_a]
    db = df[(df.id >= 5205) & (df.id <= 5215)][cols_b]
    dc = df[(df.id >= 5213) & (df.id <= 5220)][cols_c]

    return (a, b, c), (da, db, dc)


@pytest.mark.parametrize("distinct", [False, True], ids=["all", "distinct"])
@pytest.mark.broken(["druid"], raises=sa.exc.ProgrammingError)
def test_union(backend, union_subsets, distinct):
    (a, b, c), (da, db, dc) = union_subsets

    expr = ibis.union(a, b, distinct=distinct).order_by("id")
    result = expr.execute()

    expected = pd.concat([da, db], axis=0).sort_values("id").reset_index(drop=True)
    if distinct:
        expected = expected.drop_duplicates("id")

    backend.assert_frame_equal(result, expected)


@pytest.mark.broken(["druid"], raises=sa.exc.ProgrammingError)
def test_union_mixed_distinct(backend, union_subsets):
    (a, b, c), (da, db, dc) = union_subsets

    expr = a.union(b, distinct=True).union(c, distinct=False).order_by("id")
    result = expr.execute()
    expected = pd.concat(
        [pd.concat([da, db], axis=0).drop_duplicates("id"), dc], axis=0
    ).sort_values("id")

    backend.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "distinct",
    [
        param(
            False,
            marks=pytest.mark.notyet(
                ["bigquery", "dask", "pandas", "sqlite", "snowflake", "mssql"],
                reason="backend doesn't support INTERSECT ALL",
            ),
            id="all",
        ),
        param(True, id="distinct"),
    ],
)
@pytest.mark.notimpl(["polars"])
@pytest.mark.notyet(["impala"])
@pytest.mark.broken(["druid"], raises=sa.exc.ProgrammingError)
def test_intersect(backend, alltypes, df, distinct):
    a = alltypes.filter((_.id >= 5200) & (_.id <= 5210))
    b = alltypes.filter((_.id >= 5205) & (_.id <= 5215))
    c = alltypes.filter((_.id >= 5195) & (_.id <= 5208))

    # Reset index to ensure simple RangeIndex, needed for computing `expected`
    df = df.reset_index(drop=True)
    da = df[(df.id >= 5200) & (df.id <= 5210)]
    db = df[(df.id >= 5205) & (df.id <= 5215)]
    dc = df[(df.id >= 5195) & (df.id <= 5208)]

    expr = ibis.intersect(a, b, c, distinct=distinct).order_by("id")
    result = expr.execute()

    index = da.index.intersection(db.index).intersection(dc.index)
    expected = df.iloc[index].sort_values("id").reset_index(drop=True)
    if distinct:
        expected = expected.drop_duplicates()

    backend.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "distinct",
    [
        param(
            False,
            marks=pytest.mark.notyet(
                ["bigquery", "dask", "pandas", "sqlite", "snowflake", "mssql"],
                reason="backend doesn't support EXCEPT ALL",
            ),
            id="all",
        ),
        param(True, id="distinct"),
    ],
)
@pytest.mark.notimpl(["polars"])
@pytest.mark.notyet(["impala"])
@pytest.mark.broken(["druid"], raises=sa.exc.ProgrammingError)
def test_difference(backend, alltypes, df, distinct):
    a = alltypes.filter((_.id >= 5200) & (_.id <= 5210))
    b = alltypes.filter((_.id >= 5205) & (_.id <= 5215))
    c = alltypes.filter((_.id >= 5195) & (_.id <= 5202))

    # Reset index to ensure simple RangeIndex, needed for computing `expected`
    df = df.reset_index(drop=True)
    da = df[(df.id >= 5200) & (df.id <= 5210)]
    db = df[(df.id >= 5205) & (df.id <= 5215)]
    dc = df[(df.id >= 5195) & (df.id <= 5202)]

    expr = ibis.difference(a, b, c, distinct=distinct).order_by("id")
    result = expr.execute()

    index = da.index.difference(db.index).difference(dc.index)
    expected = df.iloc[index].sort_values("id").reset_index(drop=True)
    if distinct:
        expected = expected.drop_duplicates()

    backend.assert_frame_equal(result, expected)


@pytest.mark.parametrize("method", ["intersect", "difference", "union"])
def test_table_set_operations_api(alltypes, method):
    # top level variadic
    result = getattr(ibis, method)(alltypes)
    assert result.equals(alltypes)

    # table level methods require at least one argument
    with pytest.raises(
        TypeError, match="missing 1 required positional argument: 'table'"
    ):
        getattr(ir.Table, method)(alltypes)


@pytest.mark.parametrize(
    "distinct",
    [
        param(
            True, marks=pytest.mark.broken(["druid"], raises=sa.exc.ProgrammingError)
        ),
        False,
    ],
)
def test_top_level_union(backend, con, alltypes, distinct):
    t1 = alltypes.select(a="bigint_col").filter(lambda t: t.a == 10).distinct()
    t2 = alltypes.select(a="bigint_col").filter(lambda t: t.a == 20).distinct()
    expr = t1.union(t2, distinct=distinct).limit(2)
    result = con.execute(expr)
    expected = pd.DataFrame({"a": [10, 20]})
    backend.assert_frame_equal(result.sort_values("a").reset_index(drop=True), expected)


@pytest.mark.parametrize(
    "distinct",
    [
        True,
        param(
            False,
            marks=pytest.mark.notimpl(
                ["bigquery", "dask", "mssql", "pandas", "snowflake", "sqlite"]
            ),
        ),
    ],
)
@pytest.mark.parametrize(
    ("opname", "expected"),
    [
        ("intersect", pd.DataFrame({"a": [20]})),
        ("difference", pd.DataFrame({"a": [10]})),
    ],
    ids=["intersect", "difference"],
)
@pytest.mark.notimpl(["polars"], raises=com.OperationNotDefinedError)
@pytest.mark.notyet(["impala"], reason="doesn't support intersection or difference")
@pytest.mark.broken(["druid"], raises=sa.exc.ProgrammingError)
def test_top_level_intersect_difference(
    backend, con, alltypes, distinct, opname, expected
):
    t1 = (
        alltypes.select(a="bigint_col")
        .filter(lambda t: (t.a == 10) | (t.a == 20))
        .distinct()
    )
    t2 = (
        alltypes.select(a="bigint_col")
        .filter(lambda t: (t.a == 20) | (t.a == 30))
        .distinct()
    )
    op = getattr(t1, opname)
    expr = op(t2, distinct=distinct).limit(2)
    result = con.execute(expr)
    backend.assert_frame_equal(result, expected)
