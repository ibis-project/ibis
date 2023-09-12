from __future__ import annotations

import collections

import numpy as np
import pandas as pd
import pandas.testing as tm
import pytest

import ibis
import ibis.expr.datatypes as dt
import ibis.expr.types as ir
from ibis.legacy.udf.vectorized import analytic, elementwise, reduction

dd = pytest.importorskip("dask.dataframe")


@pytest.fixture
def df(npartitions):
    return dd.from_pandas(
        pd.DataFrame(
            {
                "a": list("abc"),
                "b": [1, 2, 3],
                "c": [4.0, 5.0, 6.0],
                "key": list("aab"),
            }
        ),
        npartitions=npartitions,
    )


@pytest.fixture
def df2(npartitions):
    # df with some randomness
    return dd.from_pandas(
        pd.DataFrame(
            {
                "a": np.arange(4, dtype=float).tolist() + np.random.rand(3).tolist(),
                "b": np.arange(4, dtype=float).tolist() + np.random.rand(3).tolist(),
                "c": np.arange(7, dtype=int).tolist(),
                "d": list("aaaaddd"),
                "key": list("ddeefff"),
            }
        ),
        npartitions=npartitions,
    )


@pytest.fixture
def df_timestamp(npartitions):
    df = pd.DataFrame(
        {
            "a": list(range(10)),
            "b": list("wwwwwxxxxx"),
            "c": list("yyyzzzyyzz"),
        }
    )
    df["a"] = df.a.astype(pd.DatetimeTZDtype(tz="UTC"))
    return dd.from_pandas(
        df,
        npartitions=npartitions,
    )


@pytest.fixture
def con(df, df2, df_timestamp):
    return ibis.dask.connect({"df": df, "df2": df2, "df_timestamp": df_timestamp})


@pytest.fixture
def t(con):
    return con.table("df")


@pytest.fixture
def t2(con):
    return con.table("df2")


@pytest.fixture
def t_timestamp(con):
    return con.table("df_timestamp")


# -------------
# UDF Functions
# -------------


@elementwise(input_type=["string"], output_type="int64")
def my_string_length(series, **kwargs):
    return series.str.len() * 2


@elementwise(input_type=[dt.double, dt.double], output_type=dt.double)
def my_add(series1, series2, **kwargs):
    return series1 + series2


@reduction(["double"], "double")
def my_mean(series):
    return series.mean()


@reduction(
    input_type=[dt.Timestamp(timezone="UTC")],
    output_type=dt.Timestamp(timezone="UTC"),
)
def my_tz_min(series):
    return series.min()


@elementwise(
    input_type=[dt.Timestamp(timezone="UTC")],
    output_type=dt.Timestamp(timezone="UTC"),
)
def my_tz_add_one(series):
    return series + pd.Timedelta(1, unit="D")


@reduction(input_type=[dt.string], output_type=dt.int64)
def my_string_length_sum(series, **kwargs):
    return (series.str.len() * 2).sum()


@reduction(input_type=[dt.double, dt.double], output_type=dt.double)
def my_corr(lhs, rhs, **kwargs):
    return lhs.corr(rhs)


@elementwise([dt.double], dt.double)
def add_one(x):
    return x + 1.0


@elementwise([dt.double], dt.double)
def times_two(x):
    return x * 2.0


@analytic(input_type=["double"], output_type="double")
def zscore(series):
    return (series - series.mean()) / series.std()


@reduction(
    input_type=[dt.double],
    output_type=dt.Array(dt.double),
)
def collect(series):
    return list(series)


# -----
# Tests
# -----


def test_udf(t, df):
    expr = my_string_length(t.a)

    assert isinstance(expr, ir.Column)

    result = expr.execute()
    expected = df.a.str.len().mul(2).compute()

    tm.assert_series_equal(result, expected, check_names=False, check_index=False)


def test_multiple_argument_udf(t, df):
    expr = my_add(t.b, t.c)

    assert isinstance(expr, ir.Column)
    assert isinstance(expr, ir.NumericColumn)
    assert isinstance(expr, ir.FloatingColumn)

    result = expr.execute()
    expected = (df.b + df.c).compute()
    tm.assert_series_equal(result, expected, check_index=False, check_names=False)


def test_multiple_argument_udf_group_by(t):
    expr = t.group_by(t.key).aggregate(my_add=my_add(t.b, t.c).sum())

    assert isinstance(expr, ir.Table)
    assert isinstance(expr.my_add, ir.Column)
    assert isinstance(expr.my_add, ir.NumericColumn)
    assert isinstance(expr.my_add, ir.FloatingColumn)

    result = expr.execute()
    expected = pd.DataFrame(
        {"key": list("ab"), "my_add": [sum([1.0 + 4.0, 2.0 + 5.0]), 3.0 + 6.0]}
    )
    tm.assert_frame_equal(
        result.reset_index(drop=True), expected.reset_index(drop=True)
    )


def test_udaf(t):
    expr = my_string_length_sum(t.a)

    assert isinstance(expr, ir.Scalar)

    result = expr.execute()
    expected = t.a.execute().str.len().mul(2).sum()
    assert result == expected


def test_udaf_analytic_tzcol(t_timestamp, df_timestamp):
    expr = my_tz_min(t_timestamp.a)

    result = expr.execute()

    expected = my_tz_min.func(df_timestamp.a.compute())
    assert result == expected


def test_udaf_elementwise_tzcol(t_timestamp, df_timestamp):
    expr = my_tz_add_one(t_timestamp.a)

    result = expr.execute().reset_index(drop=True)

    expected = my_tz_add_one.func(df_timestamp.a.compute())
    tm.assert_series_equal(result, expected, check_index=False, check_names=False)


def test_udaf_analytic(t, df):
    expr = zscore(t.c)

    assert isinstance(expr, ir.Column)

    result = expr.execute()

    def f(s):
        return s.sub(s.mean()).div(s.std())

    expected = (f(df.c)).compute()
    tm.assert_series_equal(result, expected, check_index=False, check_names=False)


def test_udaf_analytic_group_by(t, df):
    expr = zscore(t.c).over(ibis.window(group_by=t.key))

    assert isinstance(expr, ir.Column)

    result = expr.execute()

    def f(s):
        return s.sub(s.mean()).div(s.std())

    expected = df.groupby("key").c.transform(f).compute()
    # We don't check names here because the udf is used "directly".
    # We could potentially special case this and set the name directly
    # if the udf is only being run on one column.
    tm.assert_series_equal(
        result.sort_index(), expected.sort_index(), check_names=False, check_index=False
    )


def test_udaf_group_by(t2, df2):
    expr = t2.group_by(t2.key).aggregate(my_corr=my_corr(t2.a, t2.b))

    result = expr.execute().sort_values("key").reset_index(drop=True)

    dfi = df2.set_index("key").compute()
    expected = pd.DataFrame(
        {
            "key": list("def"),
            "my_corr": [
                dfi.loc[value, "a"].corr(dfi.loc[value, "b"]) for value in "def"
            ],
        }
    )

    tm.assert_frame_equal(
        result.reset_index(drop=True), expected.reset_index(drop=True)
    )


def test_udaf_group_by_multikey(t2, df2):
    expr = t2.group_by([t2.key, t2.d]).aggregate(my_corr=my_corr(t2.a, t2.b))

    result = expr.execute().sort_values("key").reset_index(drop=True)

    dfi = df2.set_index("key").compute()
    expected = pd.DataFrame(
        {
            "key": list("def"),
            "d": list("aad"),
            "my_corr": [
                dfi.loc[value, "a"].corr(dfi.loc[value, "b"]) for value in "def"
            ],
        }
    )
    tm.assert_frame_equal(
        result.reset_index(drop=True), expected.reset_index(drop=True)
    )


def test_udaf_group_by_multikey_tzcol(t_timestamp, df_timestamp):
    expr = t_timestamp.group_by([t_timestamp.b, t_timestamp.c]).aggregate(
        my_min_time=my_tz_min(t_timestamp.a)
    )

    result = expr.execute().sort_values("b").reset_index(drop=True)
    expected = (
        df_timestamp.groupby(["b", "c"])
        .min()
        .reset_index()
        .rename(columns={"a": "my_min_time"})
        .compute()
    )
    tm.assert_frame_equal(
        result.reset_index(drop=True), expected.reset_index(drop=True)
    )


def test_compose_udfs(t2, df2):
    expr = times_two(add_one(t2.a))
    result = expr.execute().reset_index(drop=True)
    expected = df2.a.add(1.0).mul(2.0).compute()
    tm.assert_series_equal(result, expected, check_names=False, check_index=False)


@pytest.mark.xfail(raises=NotImplementedError, reason="TODO - windowing - #2553")
def test_udaf_window(t2, df2):
    window = ibis.trailing_window(2, order_by="a", group_by="key")
    expr = t2.mutate(rolled=my_mean(t2.b).over(window))
    result = expr.execute().sort_values(["key", "a"])
    expected = df2.sort_values(["key", "a"]).assign(
        rolled=lambda df: df.groupby("key")
        .b.rolling(3, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    tm.assert_frame_equal(
        result.reset_index(drop=True), expected.reset_index(drop=True)
    )


@pytest.mark.xfail(raises=NotImplementedError, reason="TODO - windowing - #2553")
def test_udaf_window_interval(npartitions):
    df = pd.DataFrame(
        collections.OrderedDict(
            [
                (
                    "time",
                    pd.date_range(start="20190105", end="20190101", freq="-1D"),
                ),
                ("key", [1, 2, 1, 2, 1]),
                ("value", np.arange(5)),
            ]
        )
    )
    df = dd.from_pandas(df, npartitions=npartitions)

    con = ibis.dask.connect({"df": df})
    t = con.table("df")
    window = ibis.trailing_range_window(
        ibis.interval(days=2), order_by="time", group_by="key"
    )

    expr = t.mutate(rolled=my_mean(t.value).over(window))

    result = expr.execute().sort_values(["time", "key"]).reset_index(drop=True)
    expected = (
        df.sort_values(["time", "key"])
        .set_index("time")
        .assign(
            rolled=lambda df: df.groupby("key")
            .value.rolling("2D", closed="both")
            .mean()
            .reset_index(level=0, drop=True)
        )
    ).reset_index(drop=False)

    tm.assert_frame_equal(
        result.reset_index(drop=True), expected.reset_index(drop=True)
    )


@pytest.mark.xfail(raises=NotImplementedError, reason="TODO - windowing - #2553")
def test_multiple_argument_udaf_window(npartitions):
    @reduction(["double", "double"], "double")
    def my_wm(v, w):
        return np.average(v, weights=w)

    df = pd.DataFrame(
        {
            "a": np.arange(4, 0, dtype=float, step=-1).tolist()
            + np.random.rand(3).tolist(),
            "b": np.arange(4, dtype=float).tolist() + np.random.rand(3).tolist(),
            "c": np.arange(4, dtype=float).tolist() + np.random.rand(3).tolist(),
            "d": np.repeat(1, 7),
            "key": list("deefefd"),
        }
    )
    df = dd.from_pandas(df, npartitions=npartitions)

    con = ibis.dask.connect({"df": df})
    t = con.table("df")
    window = ibis.trailing_window(2, order_by="a", group_by="key")
    window2 = ibis.trailing_window(1, order_by="b", group_by="key")
    expr = t.mutate(
        wm_b=my_wm(t.b, t.d).over(window),
        wm_c=my_wm(t.c, t.d).over(window),
        wm_c2=my_wm(t.c, t.d).over(window2),
    )
    result = expr.execute().sort_values(["key", "a"])
    expected = (
        df.sort_values(["key", "a"])
        .assign(
            wm_b=lambda df: df.groupby("key")
            .b.rolling(3, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        .assign(
            wm_c=lambda df: df.groupby("key")
            .c.rolling(3, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
    )
    expected = expected.sort_values(["key", "b"]).assign(
        wm_c2=lambda df: df.groupby("key")
        .c.rolling(2, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    expected = expected.sort_values(["key", "a"])

    tm.assert_frame_equal(
        result.reset_index(drop=True), expected.reset_index(drop=True)
    )


@pytest.mark.xfail(raises=NotImplementedError, reason="TODO - windowing - #2553")
def test_udaf_window_nan(npartitions):
    df = pd.DataFrame(
        {
            "a": np.arange(10, dtype=float),
            "b": [3.0, np.NaN] * 5,
            "key": list("ddeefffggh"),
        }
    )
    df = dd.from_pandas(df, npartitions=npartitions)

    con = ibis.dask.connect({"df": df})
    t = con.table("df")
    window = ibis.trailing_window(2, order_by="a", group_by="key")
    expr = t.mutate(rolled=my_mean(t.b).over(window))
    result = expr.execute().sort_values(["key", "a"])
    expected = df.sort_values(["key", "a"]).assign(
        rolled=lambda d: d.groupby("key")
        .b.rolling(3, min_periods=1)
        .apply(lambda x: x.mean(), raw=True)
        .reset_index(level=0, drop=True)
    )
    tm.assert_frame_equal(
        result.reset_index(drop=True), expected.reset_index(drop=True)
    )


def test_array_return_type_reduction(t, df):
    """Tests reduction UDF returning an array."""
    expr = collect(t.b)
    result = expr.execute()
    expected = df.b.compute().tolist()
    assert list(result) == expected


def test_array_return_type_reduction_window(t, df):
    """Tests reduction UDF returning an array, used over a window."""
    expr = collect(t.b).over(ibis.window())
    result = expr.execute()
    expected_raw = df.b.compute().tolist()
    expected = pd.Series([expected_raw] * len(df))
    tm.assert_series_equal(result, expected, check_index=False, check_names=False)


def test_array_return_type_reduction_group_by(t, df):
    """Tests reduction UDF returning an array, used in a grouped agg."""
    expr = t.group_by(t.key).aggregate(quantiles_col=collect(t.b))
    result = expr.execute()

    df = df.compute()  # Convert to Pandas
    expected_col = df.groupby(df.key).b.agg(lambda s: s.tolist())
    expected = pd.DataFrame({"quantiles_col": expected_col}).reset_index()

    tm.assert_frame_equal(
        result.sort_values("key").reset_index(drop=True),
        expected.sort_values("key").reset_index(drop=True),
    )


def test_elementwise_udf_with_many_args(t2):
    @elementwise(input_type=[dt.double] * 16 + [dt.int32] * 8, output_type=dt.double)
    def my_udf(
        c1,
        c2,
        c3,
        c4,
        c5,
        c6,
        c7,
        c8,
        c9,
        c10,
        c11,
        c12,
        c13,
        c14,
        c15,
        c16,
        c17,
        c18,
        c19,
        c20,
        c21,
        c22,
        c23,
        c24,
    ):
        return c1

    expr = my_udf(*([t2.a] * 8 + [t2.b] * 8 + [t2.c] * 8))
    result = expr.execute()
    expected = t2.a.execute()

    tm.assert_series_equal(result, expected, check_names=False, check_index=False)


# -----------------
# Test raied errors
# -----------------


def test_udaf_parameter_mismatch():
    with pytest.raises(TypeError):

        @reduction(input_type=[dt.double], output_type=dt.double)
        def my_corr(lhs, rhs, **kwargs):
            pass


def test_udf_parameter_mismatch():
    with pytest.raises(TypeError):

        @reduction(input_type=[], output_type=dt.double)
        def my_corr2(lhs, **kwargs):
            pass


def test_udf_error(t):
    @elementwise(input_type=[dt.double], output_type=dt.double)
    def error_udf(s):
        raise ValueError("xxx")

    with pytest.raises(ValueError):
        error_udf(t.c).execute()
