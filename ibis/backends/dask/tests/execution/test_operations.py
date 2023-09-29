from __future__ import annotations

import operator
from operator import methodcaller

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest
from pytest import param

import ibis
import ibis.expr.datatypes as dt
from ibis.common.exceptions import OperationNotDefinedError

da = pytest.importorskip("dask.array")
dd = pytest.importorskip("dask.dataframe")

from dask.dataframe.utils import tm  # noqa: E402

from ibis.backends.dask.execution import execute  # noqa: E402


def test_table_column(t, df):
    expr = t.plain_int64
    result = expr.compile()
    expected = df.plain_int64
    tm.assert_series_equal(result.compute(), expected.compute(), check_index=False)


def test_literal(client):
    assert client.execute(ibis.literal(1)) == 1


def test_selection(t, df):
    expr = t[((t.plain_strings == "a") | (t.plain_int64 == 3)) & (t.dup_strings == "d")]
    result = expr.compile()
    expected = df[
        ((df.plain_strings == "a") | (df.plain_int64 == 3)) & (df.dup_strings == "d")
    ]
    tm.assert_frame_equal(
        result[expected.columns].compute().reset_index(drop=True),
        expected.compute().reset_index(drop=True),
    )


def test_mutate(t, df):
    expr = t.mutate(x=t.plain_int64 + 1, y=t.plain_int64 * 2)
    result = expr.compile()
    expected = df.assign(x=df.plain_int64 + 1, y=df.plain_int64 * 2)
    tm.assert_frame_equal(
        result[expected.columns].compute().reset_index(drop=True),
        expected.compute().reset_index(drop=True),
    )


@pytest.mark.xfail(reason="TODO - windowing - #2553")
def test_project_scope_does_not_override(t, df):
    col = t.plain_int64
    expr = t[
        [
            col.name("new_col"),
            col.sum().over(ibis.window(group_by="dup_strings")).name("grouped"),
        ]
    ]
    result = expr.compile()
    expected = dd.concat(
        [
            df[["plain_int64", "dup_strings"]].rename(
                columns={"plain_int64": "new_col"}
            ),
            df.groupby("dup_strings")
            .plain_int64.transform("sum")
            .reset_index(drop=True)
            .rename("grouped"),
        ],
        axis=1,
    )[["new_col", "grouped"]]
    tm.assert_frame_equal(
        result.compute().reset_index(drop=True),
        expected.compute().reset_index(drop=True),
    )


@pytest.mark.parametrize(
    "where",
    [
        lambda t: None,
        lambda t: t.dup_strings == "d",
        lambda t: (t.dup_strings == "d") | (t.plain_int64 < 100),
    ],
)
@pytest.mark.parametrize(
    ("ibis_func", "dask_func"),
    [
        (methodcaller("abs"), np.abs),
        (methodcaller("ceil"), np.ceil),
        (methodcaller("exp"), np.exp),
        (methodcaller("floor"), np.floor),
        (methodcaller("ln"), np.log),
        (methodcaller("log10"), np.log10),
        (methodcaller("log", 2), lambda x: np.log(x) / np.log(2)),
        (methodcaller("log2"), np.log2),
        (methodcaller("round", 0), lambda x: x.round(0).astype("int64")),
        (methodcaller("round", -2), methodcaller("round", -2)),
        (methodcaller("round", 2), methodcaller("round", 2)),
        (methodcaller("round"), lambda x: x.round().astype("int64")),
        (methodcaller("sign"), np.sign),
        (methodcaller("sqrt"), np.sqrt),
    ],
)
def test_aggregation_group_by(t, df, where, ibis_func, dask_func):
    ibis_where = where(t)
    expr = t.group_by(t.dup_strings).aggregate(
        avg_plain_int64=t.plain_int64.mean(where=ibis_where),
        sum_plain_float64=t.plain_float64.sum(where=ibis_where),
        mean_float64_positive=ibis_func(t.float64_positive).mean(where=ibis_where),
        neg_mean_int64_with_zeros=(-t.int64_with_zeros).mean(where=ibis_where),
        nunique_dup_ints=t.dup_ints.nunique(),
    )
    result = expr.compile()

    dask_where = where(df.compute())
    mask = slice(None) if dask_where is None else dask_where
    expected = (
        df.compute()
        .groupby("dup_strings")
        .agg(
            {
                "plain_int64": lambda x, mask=mask: x[mask].mean(),
                # Note we force min count here to match dask behavior
                "plain_float64": lambda x, mask=mask: x[mask].sum(min_count=1),
                "dup_ints": "nunique",
                "float64_positive": (
                    lambda x, mask=mask, func=dask_func: func(x[mask]).mean()
                ),
                "int64_with_zeros": lambda x, mask=mask: (-x[mask]).mean(),
            }
        )
        .reset_index()
        .rename(
            columns={
                "plain_int64": "avg_plain_int64",
                "plain_float64": "sum_plain_float64",
                "dup_ints": "nunique_dup_ints",
                "float64_positive": "mean_float64_positive",
                "int64_with_zeros": "neg_mean_int64_with_zeros",
            }
        )
    )

    result = result.compute()

    # TODO(phillipc): Why does pandas not return floating point values here?
    expected["avg_plain_int64"] = expected.avg_plain_int64.astype("float64")
    result["avg_plain_int64"] = result.avg_plain_int64.astype("float64")
    expected["neg_mean_int64_with_zeros"] = expected.neg_mean_int64_with_zeros.astype(
        "float64"
    )
    result["neg_mean_int64_with_zeros"] = result.neg_mean_int64_with_zeros.astype(
        "float64"
    )
    expected["mean_float64_positive"] = expected.mean_float64_positive.astype("float64")
    result["mean_float64_positive"] = result.mean_float64_positive.astype("float64")

    lhs = result[expected.columns]
    rhs = expected
    tm.assert_frame_equal(lhs, rhs)


def test_aggregation_without_group_by(t, df):
    expr = t.aggregate(
        avg_plain_int64=t.plain_int64.mean(),
        sum_plain_float64=t.plain_float64.sum(),
    )
    result = expr.compile()[["avg_plain_int64", "sum_plain_float64"]]
    new_names = {
        "plain_float64": "sum_plain_float64",
        "plain_int64": "avg_plain_int64",
    }
    pandas_df = df.compute().reset_index(drop=True)
    expected = (
        pd.Series(
            [
                pandas_df["plain_int64"].mean(),
                pandas_df["plain_float64"].sum(),
            ],
            index=["plain_int64", "plain_float64"],
        )
        .to_frame()
        .T.rename(columns=new_names)
    )
    lhs = result[expected.columns].compute().reset_index(drop=True)
    tm.assert_frame_equal(lhs, expected)


def test_group_by_with_having(t, df):
    expr = (
        t.group_by(t.dup_strings)
        .having(t.plain_float64.sum() == 5)
        .aggregate(avg_a=t.plain_int64.mean(), sum_c=t.plain_float64.sum())
    )
    result = expr.compile()

    expected = (
        df.groupby("dup_strings")
        .agg({"plain_int64": "mean", "plain_float64": "sum"})
        .reset_index()
        .rename(columns={"plain_int64": "avg_a", "plain_float64": "sum_c"})
    )
    expected = expected.loc[expected.sum_c == 5, ["avg_a", "sum_c"]]

    tm.assert_frame_equal(
        result[expected.columns].compute().reset_index(drop=True),
        expected.compute().reset_index(drop=True),
    )


def test_group_by_rename_key(t, df):
    expr = t.group_by(t.dup_strings.name("foo")).aggregate(
        dup_string_count=t.dup_strings.count()
    )

    assert "foo" in expr.schema()
    result = expr.compile()
    assert "foo" in result.columns

    expected = (
        df.groupby("dup_strings")
        .dup_strings.count()
        .rename("dup_string_count")
        .reset_index()
        .rename(columns={"dup_strings": "foo"})
    )
    tm.assert_frame_equal(
        result.compute().reset_index(drop=True),
        expected.compute().reset_index(drop=True),
    )


@pytest.mark.parametrize("reduction", ["mean", "sum", "count", "std", "var"])
@pytest.mark.parametrize(
    "where",
    [
        lambda t: (t.plain_strings == "a") | (t.plain_strings == "c"),
        lambda t: (t.dup_strings == "d")
        & ((t.plain_int64 == 1) | (t.plain_int64 == 3)),
        lambda t: None,
    ],
)
def test_reduction(t, df, reduction, where):
    func = getattr(t.plain_int64, reduction)
    mask = where(t)
    expr = func(where=mask)
    result = expr.compile()

    df_mask = where(df)
    expected_func = getattr(
        df.loc[df_mask if df_mask is not None else slice(None), "plain_int64"],
        reduction,
    )
    expected = expected_func()
    assert result.compute() == expected.compute()


@pytest.mark.parametrize(
    "where",
    [
        lambda t: (t.plain_strings == "a") | (t.plain_strings == "c"),
        lambda t: None,
    ],
)
def test_grouped_reduction(t, df, where):
    ibis_where = where(t)
    expr = t.group_by(t.dup_strings).aggregate(
        nunique_dup_ints=t.dup_ints.nunique(),
        sum_plain_int64=t.plain_int64.sum(where=ibis_where),
        mean_plain_int64=t.plain_int64.mean(where=ibis_where),
        count_plain_int64=t.plain_int64.count(where=ibis_where),
        std_plain_int64=t.plain_int64.std(where=ibis_where),
        var_plain_int64=t.plain_int64.var(where=ibis_where),
        nunique_plain_int64=t.plain_int64.nunique(where=ibis_where),
    )
    result = expr.compile()

    df_mask = where(df.compute())
    mask = slice(None) if df_mask is None else df_mask

    expected = (
        df.compute()
        .groupby("dup_strings")
        .agg(
            {
                "dup_ints": "nunique",
                "plain_int64": [
                    lambda x, mask=mask: x[mask].sum(),
                    lambda x, mask=mask: x[mask].mean(),
                    lambda x, mask=mask: x[mask].count(),
                    lambda x, mask=mask: x[mask].std(),
                    lambda x, mask=mask: x[mask].var(),
                    lambda x, mask=mask: x[mask].nunique(),
                ],
            }
        )
        .reset_index()
    )
    result = result.compute()

    assert len(result.columns) == len(expected.columns)

    expected.columns = [
        "dup_strings",
        "nunique_dup_ints",
        "sum_plain_int64",
        "mean_plain_int64",
        "count_plain_int64",
        "std_plain_int64",
        "var_plain_int64",
        "nunique_plain_int64",
    ]
    # guarantee ordering
    result = result[expected.columns]
    # dask and pandas differ slightly in how they treat groups with no entry
    # we're not testing that so fillna here.
    result = result.fillna(0.0)
    expected = expected.fillna(0.0)

    # match the dtypes
    if df_mask is None:
        expected["mean_plain_int64"] = expected.mean_plain_int64.astype("float64")
    else:
        expected["sum_plain_int64"] = expected.sum_plain_int64.astype("float64")
        expected["count_plain_int64"] = expected.count_plain_int64.astype("float64")
        expected["nunique_plain_int64"] = expected.nunique_plain_int64.astype("float64")

    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "reduction",
    [
        lambda x: x.any(),
        lambda x: x.all(),
        lambda x: ~(x.any()),
        lambda x: ~(x.all()),
    ],
)
def test_boolean_aggregation(t, df, reduction):
    expr = reduction(t.plain_int64 == 1)
    result = expr.compile()
    expected = reduction(df.plain_int64 == 1)
    assert result.compute() == expected.compute()


@pytest.mark.parametrize("column", ["float64_with_zeros", "int64_with_zeros"])
def test_null_if_zero(t, df, column):
    with pytest.warns(FutureWarning):
        expr = t[column].nullifzero()
    result = expr.compile()
    expected = df[column].replace(0, np.nan)
    tm.assert_series_equal(result.compute(), expected.compute(), check_index=False)


@pytest.mark.parametrize(
    ("left", "right", "expected", "compare"),
    [
        param(
            lambda t: ibis.literal(1),
            lambda t: ibis.literal(1),
            lambda df: np.nan,
            np.testing.assert_array_equal,  # treats NaNs as equal
            id="literal_literal_equal",
        ),
        param(
            lambda t: ibis.literal(1),
            lambda t: ibis.literal(2),
            lambda df: 1,
            np.testing.assert_equal,
            id="literal_literal_not_equal",
        ),
        param(
            lambda t: t.dup_strings,
            lambda t: ibis.literal("a"),
            lambda df: df.dup_strings.where(df.dup_strings != "a"),
            tm.assert_series_equal,
            id="series_literal",
        ),
        param(
            lambda t: t.dup_strings,
            lambda t: t.dup_strings,
            lambda df: df.dup_strings.where(df.dup_strings != df.dup_strings),
            tm.assert_series_equal,
            id="series_series",
        ),
        param(
            lambda _: ibis.literal("a"),
            lambda t: t.dup_strings,
            lambda _: dd.from_array(
                np.array(["d", np.nan, "d"], dtype="object")
            ).rename("dup_strings"),
            tm.assert_series_equal,
            id="literal_series",
        ),
    ],
)
def test_nullif(t, df, left, right, expected, compare):
    expr = left(t).nullif(right(t))
    result = execute(expr.op())
    if isinstance(result, (dd.Series, dd.DataFrame)):
        compare(
            result.compute().reset_index(drop=True),
            expected(df).compute().reset_index(drop=True),
        )
    else:
        compare(result, expected(df))


def test_nullif_inf(npartitions):
    df = dd.from_pandas(
        pd.DataFrame({"a": [np.inf, 3.14, -np.inf, 42.0]}),
        npartitions=npartitions,
    )
    con = ibis.dask.connect({"t": df})
    t = con.table("t")
    expr = t.a.nullif(np.inf).nullif(-np.inf)
    result = expr.compile()
    expected = dd.from_pandas(
        pd.Series([np.nan, 3.14, np.nan, 42.0], name="a"),
        npartitions=npartitions,
    ).reset_index(
        drop=True
    )  # match dask reset index behavior
    tm.assert_series_equal(result.compute(), expected.compute(), check_index=False)


def test_group_concat(t, df):
    expr = (
        t[t.dup_ints == 1]
        .group_by(t.dup_strings)
        .aggregate(foo=t.dup_ints.group_concat(","))
    )
    result = expr.compile()
    expected = (
        df[df.dup_ints == 1]
        .groupby("dup_strings")
        .apply(lambda df: ",".join(df.dup_ints.astype(str)))
        .reset_index()
        .rename(columns={0: "foo"})
    )

    left = (
        result[expected.columns]
        .compute()
        .sort_values("dup_strings")
        .reset_index(drop=True)
    )
    right = expected.compute().sort_values("dup_strings").reset_index(drop=True)
    tm.assert_frame_equal(left, right)


@pytest.mark.parametrize("offset", [0, 2])
def test_frame_limit(t, df, offset):
    n = 5
    df_expr = t.limit(n, offset=offset)
    result = df_expr.compile()
    expected = df.loc[offset : offset + n].reset_index(drop=True)
    tm.assert_frame_equal(
        result[expected.columns].compute().reset_index(drop=True),
        expected.compute().reset_index(drop=True),
    )


@pytest.mark.xfail(raises=AttributeError, reason="TableColumn does not implement limit")
@pytest.mark.parametrize("offset", [0, 2])
def test_series_limit(t, df, offset):
    n = 5
    s_expr = t.plain_int64.limit(n, offset=offset)
    result = s_expr.compile()
    tm.assert_series_equal(
        result, df.plain_int64.iloc[offset : offset + n], check_index=False
    )


@pytest.mark.parametrize(
    ("key", "dask_by", "dask_ascending"),
    [
        (lambda t, col: [t[col]], lambda col: [col], True),
        param(
            lambda t, col: [ibis.desc(t[col])],
            lambda col: [col],
            False,
            marks=pytest.mark.xfail(reason="TODO -sorting - #2553"),
        ),
        param(
            lambda t, col: [t[col], t.plain_int64],
            lambda col: [col, "plain_int64"],
            [True, False],
            marks=pytest.mark.xfail(reason="TODO - sorting - #2553"),
        ),
        (
            lambda t, col: [t.plain_int64 * 2],
            lambda col: ["plain_int64"],
            True,
        ),
    ],
)
@pytest.mark.parametrize(
    "column",
    ["plain_datetimes_naive", "plain_datetimes_ny", "plain_datetimes_utc"],
)
def test_order_by(t, df, column, key, dask_by, dask_ascending):
    expr = t.order_by(key(t, column))
    result = expr.compile()
    expected = (
        df.compute()
        .sort_values(dask_by(column), ascending=dask_ascending)
        .reset_index(drop=True)
    )
    tm.assert_frame_equal(
        result[expected.columns].compute().reset_index(drop=True), expected
    )


@pytest.mark.xfail(reason="TODO - sorting - #2553")
def test_complex_order_by(t, df):
    expr = t.order_by([ibis.desc(t.plain_int64 * t.plain_float64), t.plain_float64])
    result = expr.compile()
    expected = (
        df.assign(foo=df.plain_int64 * df.plain_float64)
        .sort_values(["foo", "plain_float64"], ascending=[False, True])
        .drop(["foo"], axis=1)
        .reset_index(drop=True)
    )

    tm.assert_frame_equal(
        result[expected.columns].compute().reset_index(drop=True),
        expected.compute().reset_index(drop=True),
    )


def test_count_distinct(t, df):
    expr = t.dup_strings.nunique()
    result = expr.compile()
    expected = df.dup_strings.nunique()
    assert result.compute() == expected.compute()


def test_value_counts(t, df):
    expr = t.dup_strings.value_counts()
    result = expr.compile()
    expected = (
        df.compute()
        .dup_strings.value_counts()
        .rename("dup_strings")
        .reset_index(name="dup_strings_count")
        .rename(columns={"index": "dup_strings"})
        .sort_values(["dup_strings"])
        .reset_index(drop=True)
    )
    tm.assert_frame_equal(
        result[expected.columns].compute().reset_index(drop=True), expected
    )


def test_table_count(t, df):
    expr = t.count()
    result = expr.compile()
    expected = len(df)
    assert result == expected


def test_weighted_average(t, df):
    expr = t.group_by(t.dup_strings).aggregate(
        avg=(t.plain_float64 * t.plain_int64).sum() / t.plain_int64.sum()
    )
    result = expr.compile()
    expected = (
        df.groupby("dup_strings")
        .apply(
            lambda df: (df.plain_int64 * df.plain_float64).sum() / df.plain_int64.sum()
        )
        .reset_index()
        .rename(columns={0: "avg"})
    )
    tm.assert_frame_equal(
        result[expected.columns].compute().reset_index(drop=True),
        expected.compute().reset_index(drop=True),
    )


def test_group_by_multiple_keys(t, df):
    expr = t.group_by([t.dup_strings, t.dup_ints]).aggregate(
        avg_plain_float64=t.plain_float64.mean()
    )
    result = expr.compile()
    expected = (
        df.groupby(["dup_strings", "dup_ints"])
        .agg({"plain_float64": "mean"})
        .reset_index()
        .rename(columns={"plain_float64": "avg_plain_float64"})
    )
    tm.assert_frame_equal(
        result[expected.columns]
        .compute()
        .sort_values(["dup_strings", "dup_ints"])
        .reset_index(drop=True),
        expected.compute()
        .sort_values(["dup_strings", "dup_ints"])
        .reset_index(drop=True),
    )


def test_mutate_after_group_by(t, df):
    gb = t.group_by(t.dup_strings).aggregate(avg_plain_float64=t.plain_float64.mean())
    expr = gb.mutate(x=gb.avg_plain_float64)
    result = expr.compile()
    expected = (
        df.groupby("dup_strings")
        .agg({"plain_float64": "mean"})
        .reset_index()
        .rename(columns={"plain_float64": "avg_plain_float64"})
    )
    expected = expected.assign(x=expected.avg_plain_float64)
    tm.assert_frame_equal(
        result[expected.columns]
        .compute()
        .sort_values("dup_strings")
        .reset_index(drop=True),
        expected.compute().sort_values("dup_strings").reset_index(drop=True),
    )


def test_group_by_with_unnamed_arithmetic(t, df):
    expr = t.group_by(t.dup_strings).aggregate(
        naive_variance=((t.plain_float64**2).sum() - t.plain_float64.mean() ** 2)
        / t.plain_float64.count()
    )
    result = expr.compile()
    expected = (
        df.compute()
        .groupby("dup_strings")
        .agg({"plain_float64": lambda x: ((x**2).sum() - x.mean() ** 2) / x.count()})
        .reset_index()
        .rename(columns={"plain_float64": "naive_variance"})
    )
    tm.assert_frame_equal(
        result[expected.columns].compute().reset_index(drop=True), expected
    )


def test_isnull(t, df):
    expr = t.strings_with_nulls.isnull()
    result = expr.compile()
    expected = df.strings_with_nulls.isnull()
    tm.assert_series_equal(result.compute(), expected.compute(), check_index=False)


def test_notnull(t, df):
    expr = t.strings_with_nulls.notnull()
    result = expr.compile()
    expected = df.strings_with_nulls.notnull()
    tm.assert_series_equal(result.compute(), expected.compute(), check_index=False)


@pytest.mark.parametrize("raw_value", [0.0, 1.0])
def test_scalar_parameter(t, df, raw_value):
    value = ibis.param(dt.double)
    expr = t.float64_with_zeros == value
    result = expr.compile(params={value: raw_value})
    expected = df.float64_with_zeros == raw_value
    tm.assert_series_equal(result.compute(), expected.compute(), check_index=False)


@pytest.mark.parametrize("elements", [[1], (1,), {1}, frozenset({1})])
def test_isin(t, df, elements):
    expr = t.plain_float64.isin(elements)
    expected = df.plain_float64.isin(elements)
    result = expr.compile()
    tm.assert_series_equal(result.compute(), expected.compute(), check_index=False)


@pytest.mark.parametrize("elements", [[1], (1,), {1}, frozenset({1})])
def test_notin(t, df, elements):
    expr = t.plain_float64.notin(elements)
    expected = ~df.plain_float64.isin(elements)
    result = expr.compile()
    tm.assert_series_equal(result.compute(), expected.compute(), check_index=False)


def test_cast_on_group_by(t, df):
    expr = t.group_by(t.dup_strings).aggregate(
        casted=(t.float64_with_zeros == 0).cast("int64").sum()
    )

    result = expr.compile()
    expected = (
        df.groupby("dup_strings")
        .float64_with_zeros.apply(lambda s: (s == 0).astype("int64").sum())
        .reset_index()
        .rename(columns={"float64_with_zeros": "casted"})
    )
    tm.assert_frame_equal(
        result.compute().reset_index(drop=True),
        expected.compute().reset_index(drop=True),
    )


@pytest.mark.parametrize(
    "op",
    [
        operator.add,
        operator.mul,
        operator.sub,
        operator.truediv,
        operator.floordiv,
        operator.mod,
        operator.pow,
    ],
    ids=operator.attrgetter("__name__"),
)
@pytest.mark.parametrize("args", [lambda c: (1.0, c), lambda c: (c, 1.0)])
def test_left_binary_op(t, df, op, args):
    expr = op(*args(t.float64_with_zeros))
    result = expr.compile()
    expected = op(*args(df.float64_with_zeros))
    tm.assert_series_equal(result.compute(), expected.compute(), check_index=False)


@pytest.mark.parametrize(
    "op",
    [
        operator.add,
        operator.mul,
        operator.sub,
        operator.truediv,
        operator.floordiv,
        operator.mod,
        operator.pow,
    ],
    ids=operator.attrgetter("__name__"),
)
@pytest.mark.parametrize("argfunc", [lambda c: (1.0, c), lambda c: (c, 1.0)])
def test_left_binary_op_gb(t, df, op, argfunc):
    expr = t.group_by("dup_strings").aggregate(
        foo=op(*argfunc(t.float64_with_zeros)).sum()
    )
    result = expr.compile()
    expected = (
        df.groupby("dup_strings")
        .float64_with_zeros.apply(lambda s: op(*argfunc(s)).sum())
        .reset_index()
        .rename(columns={"float64_with_zeros": "foo"})
    )
    tm.assert_frame_equal(
        result.compute().reset_index(drop=True),
        expected.compute().reset_index(drop=True),
    )


@pytest.mark.parametrize(
    "left_f",
    [
        param(lambda e: e - 1, id="sub_one"),
        param(lambda _: 0.0, id="zero"),
        param(lambda _: None, id="none"),
    ],
)
@pytest.mark.parametrize(
    "right_f",
    [
        param(lambda e: e + 1, id="add_one"),
        param(lambda _: 1.0, id="one"),
        param(lambda _: None, id="none"),
    ],
)
def test_ifelse_series(t, df, left_f, right_f):
    col_expr = t["plain_int64"]
    result = ibis.ifelse(
        col_expr > col_expr.mean(), left_f(col_expr), right_f(col_expr)
    ).execute()

    series = df["plain_int64"].compute()
    cond = series > series.mean()
    left = left_f(series)
    if not isinstance(left, pd.Series):
        left = pd.Series(np.repeat(left, len(cond)), name=cond.name)
    expected = left.where(cond, right_f(series)).astype(result.dtype)

    tm.assert_series_equal(result, expected, check_index=False, check_names=False)


@pytest.mark.parametrize(
    ("cond", "expected_func"),
    [
        (True, lambda df: df["plain_int64"]),
        (False, lambda df: dd.from_array(np.repeat(3.0, len(df)))),
    ],
)
def test_ifelse_scalar(t, df, cond, expected_func):
    expr = ibis.ifelse(cond, t["plain_int64"], 3.0)
    result = expr.compile()
    expected = expected_func(df)
    tm.assert_series_equal(result.compute(), expected.compute(), check_index=False)


def test_ifelse_long(batting, batting_df):
    col_expr = batting["AB"]
    result = ibis.ifelse(col_expr > col_expr.mean(), col_expr, 0.0).compile()

    series = batting_df["AB"]
    expected = series.where(series > series.mean(), other=0.0)

    tm.assert_series_equal(result.compute(), expected.compute(), check_index=False)


def test_round(t, df):
    precision = 2
    mult = 3.33333
    result = (t.count() * mult).round(precision).compile()
    expected = np.around(len(df) * mult, precision)
    npt.assert_almost_equal(result, expected, decimal=precision)


@pytest.mark.xfail(
    raises=OperationNotDefinedError,
    reason="MultiQuantile is not implemented for the dask backend",
)
def test_quantile_group_by(batting, batting_df):
    def q_fun(x, quantile):
        res = x.quantile(quantile).tolist()
        return [res for _ in range(len(x))]

    frac = 0.2
    result = (
        batting.group_by("teamID")
        .mutate(res=lambda x: x.RBI.quantile([frac, 1 - frac]))
        .res.compile()
    )
    expected = (
        batting_df.groupby("teamID")
        .RBI.transform(q_fun, quantile=[frac, 1 - frac])
        .rename("res")
    )
    tm.assert_series_equal(result.compute(), expected.compute(), check_index=False)


def test_searched_case_scalar(client):
    expr = ibis.case().when(True, 1).when(False, 2).end()
    result = client.execute(expr)
    expected = np.int8(1)
    assert result == expected


def test_searched_case_column(batting, batting_df):
    t = batting
    df = batting_df
    expr = (
        ibis.case()
        .when(t.RBI < 5, "really bad team")
        .when(t.teamID == "PH1", "ph1 team")
        .else_(t.teamID)
        .end()
    )
    result = expr.compile()
    expected = dd.from_array(
        np.select(
            [df.RBI < 5, df.teamID == "PH1"],
            ["really bad team", "ph1 team"],
            df.teamID,
        )
    )
    tm.assert_series_equal(result.compute(), expected.compute(), check_index=False)


def test_simple_case_scalar(client):
    x = ibis.literal(2)
    expr = x.case().when(2, x - 1).when(3, x + 1).when(4, x + 2).end()
    result = client.execute(expr)
    expected = np.int8(1)
    assert result == expected


def test_simple_case_column(batting, batting_df):
    t = batting
    df = batting_df
    expr = (
        t.RBI.case()
        .when(5, "five")
        .when(4, "four")
        .when(3, "three")
        .else_("could be good?")
        .end()
    )
    result = expr.compile()
    expected = dd.from_array(
        np.select(
            [df.RBI == 5, df.RBI == 4, df.RBI == 3],
            ["five", "four", "three"],
            "could be good?",
        )
    )
    tm.assert_series_equal(result.compute(), expected.compute(), check_index=False)


def test_table_distinct(t, df):
    expr = t[["dup_strings"]].distinct()
    result = expr.compile()
    expected = df[["dup_strings"]].drop_duplicates()
    tm.assert_frame_equal(
        result.compute().reset_index(drop=True),
        expected.compute().reset_index(drop=True),
    )


@pytest.mark.parametrize("distinct", [True, False])
def test_union(client, df1, distinct):
    t = client.table("df1")
    expr = t.union(t, distinct=distinct)
    result = expr.compile()
    expected = df1 if distinct else dd.concat([df1, df1], axis=0, ignore_index=True)

    # match indices because of dask reset_index behavior
    result = result.compute().reset_index(drop=True)
    expected = expected.compute().reset_index(drop=True)

    tm.assert_frame_equal(result, expected)


def test_intersect(client, df1, intersect_df2):
    t1 = client.table("df1")
    t2 = client.table("intersect_df2")
    expr = t1.intersect(t2)
    result = expr.compile()
    expected = df1.merge(intersect_df2, on=list(df1.columns))
    tm.assert_frame_equal(
        result.compute().reset_index(drop=True),
        expected.compute().reset_index(drop=True),
    )


def test_difference(client, df1, intersect_df2):
    t1 = client.table("df1")
    t2 = client.table("intersect_df2")
    expr = t1.difference(t2)
    result = expr.compile()
    merged = df1.merge(intersect_df2, on=list(df1.columns), how="outer", indicator=True)
    expected = merged[merged["_merge"] != "both"].drop("_merge", axis=1)

    # force same index
    result = result.compute().reset_index(drop=True)
    expected = expected.compute().reset_index(drop=True)

    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "distinct",
    [
        param(
            True,
            marks=pytest.mark.xfail(
                raises=TypeError,
                reason="dask cannot compute the distinct element of an array column",
            ),
        ),
        False,
    ],
)
def test_union_with_list_types(t, df, distinct):
    expr = t.union(t, distinct=distinct)
    result = expr.compile()
    expected = df if distinct else dd.concat([df, df], axis=0, ignore_index=True)
    tm.assert_frame_equal(
        result.compute().reset_index(drop=True),
        expected.compute().reset_index(drop=True),
    )
