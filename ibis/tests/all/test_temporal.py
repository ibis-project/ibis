import pytest
from pytest import param

import ibis
import pandas as pd


def test_time():
    pass


def test_date():
    pass


@pytest.mark.parametrize('unit', [
    'Y', 'M', 'D',  # 'W' TODO(kszucs): seems like numpy choses wednesday for W
    'h', 'm', 's', 'ms', 'us', 'ns'
])
def test_timestamp_truncate(backend, alltypes, df, unit):
    expr = alltypes.timestamp_col.truncate(unit)

    dtype = 'datetime64[{}]'.format(unit)
    expected = pd.Series(df.timestamp_col.values.astype(dtype))

    with backend.skip_unsupported():
        result = expr.execute()

    expected = backend.default_series_rename(expected)
    backend.assert_series_equal(result, expected)


@pytest.mark.parametrize('unit', [
    'Y', 'M', 'D',  # 'W' TODO(kszucs): seems like numpy choses wednesday for W
])
def test_date_truncate(backend, alltypes, df, unit):
    expr = alltypes.timestamp_col.date().truncate(unit)

    dtype = 'datetime64[{}]'.format(unit)
    expected = pd.Series(df.timestamp_col.values.astype(dtype))

    with backend.skip_unsupported():
        result = expr.execute()

    expected = backend.default_series_rename(expected)
    backend.assert_series_equal(result, expected)


@pytest.mark.parametrize('unit', [
    param('Y', marks=pytest.mark.xfail),
    param('M', marks=pytest.mark.xfail),
    'W',
    'D',
    'h',
    'm',
    's'
])
def test_integer_to_interval(backend, con, alltypes, df, unit):
    expr = alltypes.timestamp_col + alltypes.int_col.to_interval(unit=unit)
    # The following is incorrect for Y and M
    expected = df.timestamp_col + pd.to_timedelta(df.int_col, unit=unit)

    with backend.skip_unsupported():
        result = con.execute(expr)

    expected = backend.default_series_rename(expected)
    backend.assert_series_equal(result, expected)


@pytest.mark.parametrize(('expr_fn', 'expected_fn'), [
    param(lambda t: t.timestamp_col + ibis.interval(days=4),
          lambda t: t.timestamp_col + pd.Timedelta(days=4),
          id='timestamp-add-days'),
    param(lambda t: t.timestamp_col - ibis.interval(days=4),
          lambda t: t.timestamp_col - pd.Timedelta(days=4),
          id='timestamp-sub-days'),

])
def test_timestamp_binop(backend, con, alltypes, df,
                         expr_fn, expected_fn):
    expr = expr_fn(alltypes)
    expected = expected_fn(df)

    with backend.skip_unsupported():
        result = con.execute(expr)

    expected = backend.default_series_rename(expected)
    backend.assert_series_equal(result, expected)
