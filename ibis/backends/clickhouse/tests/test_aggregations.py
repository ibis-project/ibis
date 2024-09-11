from __future__ import annotations

from operator import methodcaller

import numpy as np
import pandas.testing as tm
import pytest

import ibis
from ibis.common.annotations import ValidationError

pytest.importorskip("clickhouse_connect")


@pytest.mark.parametrize(
    "reduction", ["sum", "count", "mean", "max", "min", "std", "var"]
)
def test_reduction_where(alltypes, reduction, assert_sql):
    method = getattr(alltypes.double_col, reduction)
    cond = alltypes.bigint_col < 70
    expr = method(where=cond)

    assert_sql(expr)


@pytest.mark.parametrize("method", ["var", "std"])
def test_std_var_pop(con, alltypes, method, assert_sql):
    cond = alltypes.bigint_col < 70
    col = alltypes.double_col
    expr = getattr(col, method)(where=cond, how="pop")
    assert_sql(expr)
    assert isinstance(con.execute(expr), float)


@pytest.mark.parametrize("reduction", ["sum", "count", "max", "min"])
def test_reduction_invalid_where(alltypes, reduction):
    condbad_literal = ibis.literal("T")

    with pytest.raises(ValidationError):
        fn = methodcaller(reduction, where=condbad_literal)
        fn(alltypes.double_col)


@pytest.mark.parametrize(
    ("func", "pandas_func"),
    [
        (
            lambda t, cond: t.bool_col.count(),
            lambda df, cond: df.bool_col.count(),
        ),
        (
            lambda t, cond: t.bool_col.approx_nunique(),
            lambda df, cond: df.bool_col.nunique(),
        ),
        (
            lambda t, cond: t.double_col.sum(),
            lambda df, cond: df.double_col.sum(),
        ),
        (
            lambda t, cond: t.double_col.mean(),
            lambda df, cond: df.double_col.mean(),
        ),
        (
            lambda t, cond: t.int_col.approx_median(),
            lambda df, cond: df.int_col.median(),
        ),
        (
            lambda t, cond: t.double_col.min(),
            lambda df, cond: df.double_col.min(),
        ),
        (
            lambda t, cond: t.double_col.max(),
            lambda df, cond: df.double_col.max(),
        ),
        (
            lambda t, cond: t.double_col.var(),
            lambda df, cond: df.double_col.var(),
        ),
        (
            lambda t, cond: t.double_col.std(),
            lambda df, cond: df.double_col.std(),
        ),
        (
            lambda t, cond: t.double_col.var(how="sample"),
            lambda df, cond: df.double_col.var(ddof=1),
        ),
        (
            lambda t, cond: t.double_col.std(how="pop"),
            lambda df, cond: df.double_col.std(ddof=0),
        ),
        (
            lambda t, cond: t.bool_col.count(where=cond),
            lambda df, cond: df.bool_col[cond].count(),
        ),
        (
            lambda t, cond: t.double_col.sum(where=cond),
            lambda df, cond: df.double_col[cond].sum(),
        ),
        (
            lambda t, cond: t.double_col.mean(where=cond),
            lambda df, cond: df.double_col[cond].mean(),
        ),
        (
            lambda t, cond: t.float_col.approx_median(where=cond),
            lambda df, cond: df.float_col[cond].median(),
        ),
        (
            lambda t, cond: t.double_col.min(where=cond),
            lambda df, cond: df.double_col[cond].min(),
        ),
        (
            lambda t, cond: t.double_col.max(where=cond),
            lambda df, cond: df.double_col[cond].max(),
        ),
        (
            lambda t, cond: t.double_col.var(where=cond),
            lambda df, cond: df.double_col[cond].var(),
        ),
        (
            lambda t, cond: t.double_col.std(where=cond),
            lambda df, cond: df.double_col[cond].std(),
        ),
        (
            lambda t, cond: t.double_col.var(where=cond, how="sample"),
            lambda df, cond: df.double_col[cond].var(),
        ),
        (
            lambda t, cond: t.double_col.std(where=cond, how="pop"),
            lambda df, cond: df.double_col[cond].std(ddof=0),
        ),
    ],
)
def test_aggregations(alltypes, df, func, pandas_func):
    table = alltypes.limit(100)
    count = table.count().execute()
    df = df.head(int(count))

    cond = table.string_col.isin(["1", "7"])
    mask = cond.execute().astype("bool")
    expr = func(table, cond)

    result = expr.execute()
    expected = pandas_func(df, mask)

    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize(
    "op",
    [
        methodcaller("sum"),
        methodcaller("mean"),
        methodcaller("min"),
        methodcaller("max"),
        methodcaller("std"),
        methodcaller("var"),
    ],
)
def test_boolean_reduction(alltypes, op, df):
    result = op(alltypes.bool_col).execute()
    assert result == op(df.bool_col)


def test_anonymous_aggregate(alltypes, df):
    t = alltypes
    expr = t.filter(t.double_col > t.double_col.mean())
    result = expr.execute().set_index("id")
    expected = df[df.double_col > df.double_col.mean()].set_index("id")
    tm.assert_frame_equal(result, expected, check_like=True)
