import pytest
import datetime
from operator import methodcaller

import numpy as np
import pandas as pd
import pandas.util.testing as tm  # noqa: E402
import ibis
from ibis.expr import datatypes as dt
from ibis import literal as L  # noqa: E402
from ibis.compat import PY2


pytestmark = pytest.mark.pandas


@pytest.mark.parametrize(
    ('case_func', 'expected_func'),
    [
        (lambda v: v.strftime('%Y%m%d'), lambda vt: vt.strftime('%Y%m%d')),

        (lambda v: v.year(), lambda vt: vt.year),
        (lambda v: v.month(), lambda vt: vt.month),
        (lambda v: v.day(), lambda vt: vt.day),
        (lambda v: v.hour(), lambda vt: vt.hour),
        (lambda v: v.minute(), lambda vt: vt.minute),
        (lambda v: v.second(), lambda vt: vt.second),
        (lambda v: v.millisecond(), lambda vt: int(vt.microsecond / 1e3)),
    ] + [
        (methodcaller('strftime', pattern), methodcaller('strftime', pattern))
        for pattern in [
            '%Y%m%d %H',
            'DD BAR %w FOO "DD"',
            'DD BAR %w FOO "D',
            'DD BAR "%w" FOO "D',
            'DD BAR "%d" FOO "D',
            'DD BAR "%c" FOO "D',
            'DD BAR "%x" FOO "D',
            'DD BAR "%X" FOO "D',
        ]
    ]
)
def test_timestamp_functions(case_func, expected_func):
    v = L('2015-09-01 14:48:05.359').cast('timestamp')
    vt = datetime.datetime(
        year=2015, month=9, day=1,
        hour=14, minute=48, second=5, microsecond=359000
    )
    result = case_func(v)
    expected = expected_func(vt)
    assert ibis.pandas.execute(result) == expected


@pytest.mark.parametrize(
    'column',
    [
        'datetime_strings_naive',
        'datetime_strings_ny',
        'datetime_strings_utc',
    ]
)
def test_cast_datetime_strings_to_date(t, df, column):
    expr = t[column].cast('date')
    result = expr.execute()
    expected = pd.to_datetime(
        df[column], infer_datetime_format=True
    ).dt.normalize()
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    'column',
    [
        'datetime_strings_naive',
        'datetime_strings_ny',
        'datetime_strings_utc',
    ]
)
def test_cast_datetime_strings_to_timestamp(t, df, column):
    expr = t[column].cast('timestamp')
    result = expr.execute()
    expected = pd.to_datetime(df[column], infer_datetime_format=True)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    'column',
    [
        'plain_datetimes_naive',
        'plain_datetimes_ny',
        'plain_datetimes_utc',
    ]
)
def test_cast_integer_to_temporal_type(t, df, column):
    expr = t.plain_int64.cast(t[column].type())
    result = expr.execute()
    expected = pd.Series(
        pd.to_datetime(df.plain_int64.values, unit='ns').values,
        index=df.index,
        name='plain_int64',
    ).dt.tz_localize(t[column].type().timezone)
    tm.assert_series_equal(result, expected)


def test_cast_integer_to_date(t, df):
    expr = t.plain_int64.cast('date')
    result = expr.execute()
    expected = pd.Series(
        pd.to_datetime(df.plain_int64.values, unit='D').values,
        index=df.index,
        name='plain_int64',
    )
    tm.assert_series_equal(result, expected)


@pytest.mark.skipif(PY2, reason="not enabled on PY2")
def test_times_ops(t, df):
    result = t.plain_datetimes_naive.time().between('10:00', '10:00').execute()
    expected = np.zeros(len(df), dtype=bool)
    tm.assert_numpy_array_equal(result, expected)

    result = t.plain_datetimes_naive.time().between('01:00', '02:00').execute()
    expected = np.ones(len(df), dtype=bool)
    tm.assert_numpy_array_equal(result, expected)


@pytest.mark.parametrize(
    "tz, rconstruct",
    [('US/Eastern', np.zeros),
     ('UTC', np.ones),
     (None, np.ones)])
@pytest.mark.skipif(PY2, reason="not enabled on PY2")
def test_times_ops_with_tz(t, df, tz, rconstruct):
    result = t.plain_datetimes_utc.time().between(
        '01:00', '02:00', timezone=tz).execute()
    expected = rconstruct(len(df), dtype=bool)
    tm.assert_numpy_array_equal(result, expected)

    result = t.plain_datetimes_naive.time().between(
        '01:00', '02:00', timezone=tz).execute()
    expected = rconstruct(len(df), dtype=bool)
    tm.assert_numpy_array_equal(result, expected)

    # equivalence
    expected = np.ones(len(df), dtype=bool)
    ts = t.plain_datetimes_utc.cast(dt.Timestamp(timezone=tz))
    result = ts.time().between('01:00', '02:00').execute()
    tm.assert_numpy_array_equal(result, expected)


@pytest.mark.skipif(not PY2, reason="testing for PY2")
def test_times_ops_py2(t, df):
    with pytest.raises(ValueError):
        t.plain_datetimes_naive.time()
