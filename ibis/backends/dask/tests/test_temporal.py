from __future__ import annotations

import datetime
from operator import methodcaller

import numpy as np
import pandas as pd
import pytest
from packaging.version import parse as parse_version
from pytest import param

import ibis
from ibis import literal as L
from ibis.expr import datatypes as dt

dd = pytest.importorskip("dask.dataframe")
from dask.dataframe.utils import tm  # noqa: E402


@pytest.mark.parametrize(
    ("case_func", "expected_func"),
    [
        (lambda v: v.strftime("%Y%m%d"), lambda vt: vt.strftime("%Y%m%d")),
        (lambda v: v.year(), lambda vt: vt.year),
        (lambda v: v.month(), lambda vt: vt.month),
        (lambda v: v.day(), lambda vt: vt.day),
        (lambda v: v.hour(), lambda vt: vt.hour),
        (lambda v: v.minute(), lambda vt: vt.minute),
        (lambda v: v.second(), lambda vt: vt.second),
        (lambda v: v.millisecond(), lambda vt: int(vt.microsecond / 1e3)),
    ]
    + [
        (methodcaller("strftime", pattern), methodcaller("strftime", pattern))
        for pattern in [
            "%Y%m%d %H",
            'DD BAR %w FOO "DD"',
            'DD BAR %w FOO "D',
            'DD BAR "%w" FOO "D',
            'DD BAR "%d" FOO "D',
            'DD BAR "%c" FOO "D',
            'DD BAR "%x" FOO "D',
            'DD BAR "%X" FOO "D',
        ]
    ],
)
def test_timestamp_functions(con, case_func, expected_func):
    v = L("2015-09-01 14:48:05.359").cast("timestamp")
    vt = datetime.datetime(
        year=2015,
        month=9,
        day=1,
        hour=14,
        minute=48,
        second=5,
        microsecond=359000,
    )
    result = case_func(v)
    expected = expected_func(vt)
    assert con.execute(result) == expected


@pytest.mark.parametrize(
    "column",
    ["datetime_strings_naive", "datetime_strings_ny", "datetime_strings_utc"],
)
def test_cast_datetime_strings_to_date(t, df, column):
    expr = t[column].cast("date")
    result = expr.execute()
    df_computed = df.compute()
    expected = pd.to_datetime(df_computed[column]).map(lambda x: x.date())

    tm.assert_series_equal(
        result.reset_index(drop=True).rename("tmp"),
        expected.reset_index(drop=True).rename("tmp"),
    )


@pytest.mark.parametrize(
    "column",
    ["datetime_strings_naive", "datetime_strings_ny", "datetime_strings_utc"],
)
def test_cast_datetime_strings_to_timestamp(t, pandas_df, column):
    expr = t[column].cast(dt.Timestamp(scale=9))
    result = expr.execute()
    expected = pd.to_datetime(pandas_df[column])
    if getattr(expected.dtype, "tz", None) is not None:
        expected = expected.dt.tz_convert(None)
    tm.assert_series_equal(result, expected, check_names=False)


@pytest.mark.parametrize(
    "column",
    ["plain_datetimes_naive", "plain_datetimes_ny", "plain_datetimes_utc"],
)
def test_cast_integer_to_temporal_type(t, df, pandas_df, column):
    column_type = t[column].type()
    expr = t.plain_int64.cast(column_type)
    result = expr.execute()

    expected = pd.Series(
        pd.to_datetime(pandas_df.plain_int64.values, unit="s").values,
        index=pandas_df.index,
        name="plain_int64",
    ).dt.tz_localize(column_type.timezone)

    tm.assert_series_equal(
        result.reset_index(drop=True),
        expected.reset_index(drop=True),
        check_names=False,
    )


def test_cast_integer_to_date(t, pandas_df):
    expr = t.plain_int64.cast("date")
    result = expr.execute()
    expected = pd.Series(
        pd.to_datetime(pandas_df.plain_int64.values, unit="D").date,
        index=pandas_df.index,
        name="plain_int64",
    )
    tm.assert_series_equal(result, expected, check_names=False)


def test_times_ops(t, df):
    result = t.plain_datetimes_naive.time().between("10:00", "10:00").execute()
    expected = pd.Series(np.zeros(len(df), dtype=bool))
    tm.assert_series_equal(
        result.reset_index(drop=True),
        expected.reset_index(drop=True),
        check_names=False,
    )

    result = t.plain_datetimes_naive.time().between("01:00", "02:00").execute()
    expected = pd.Series(np.ones(len(df), dtype=bool))
    tm.assert_series_equal(
        result.reset_index(drop=True),
        expected.reset_index(drop=True),
        check_names=False,
    )


@pytest.mark.parametrize(
    ("tz", "rconstruct", "column"),
    [
        ("US/Eastern", np.ones, "plain_datetimes_utc"),
        ("US/Eastern", np.zeros, "plain_datetimes_naive"),
        ("UTC", np.ones, "plain_datetimes_utc"),
        ("UTC", np.ones, "plain_datetimes_naive"),
        (None, np.ones, "plain_datetimes_utc"),
        (None, np.ones, "plain_datetimes_naive"),
    ],
    ids=lambda x: str(getattr(x, "__name__", x)).lower().replace("/", "_"),
)
def test_times_ops_with_tz(t, df, tz, rconstruct, column):
    expected = dd.from_array(rconstruct(len(df), dtype=bool))
    time = t[column].time()
    expr = time.between("01:00", "02:00", timezone=tz)
    result = expr.execute()
    tm.assert_series_equal(
        result.reset_index(drop=True),
        expected.compute().reset_index(drop=True),
        check_names=False,
    )

    # Test that casting behavior is the same as using the timezone kwarg
    ts = t[column].cast(dt.Timestamp(timezone=tz))
    expr = ts.time().between("01:00", "02:00")
    result = expr.execute()
    tm.assert_series_equal(
        result.reset_index(drop=True),
        expected.compute().reset_index(drop=True),
        check_names=False,
    )


@pytest.mark.parametrize(
    ("op", "expected"),
    [
        param(lambda x, y: x + y, lambda x, y: x.values * 2, id="add"),
        param(lambda x, y: x - y, lambda x, y: x.values - y.values, id="sub"),
        param(lambda x, y: x * 2, lambda x, y: x.values * 2, id="mul"),
        param(
            lambda x, y: x // 2,
            lambda x, y: x.values // 2,
            id="floordiv",
            marks=pytest.mark.xfail(
                parse_version(pd.__version__) < parse_version("0.23.0"),
                raises=TypeError,
                reason=(
                    "pandas versions less than 0.23.0 do not support floor "
                    "division involving timedelta columns"
                ),
            ),
        ),
    ],
)
def test_interval_arithmetic(op, expected):
    data = pd.timedelta_range("0 days", "10 days", freq="D")
    pandas_df = pd.DataFrame({"td": data})
    con = ibis.dask.connect(
        {
            "df1": dd.from_pandas(pandas_df, npartitions=1),
            "df2": dd.from_pandas(pandas_df, npartitions=1),
        }
    )
    t1 = con.table("df1")
    expr = op(t1.td, t1.td)
    result = expr.execute()
    expected = pd.Series(expected(data, data), name=expr.get_name())

    tm.assert_series_equal(
        result.reset_index(drop=True), expected.reset_index(drop=True)
    )
