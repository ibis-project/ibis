import pytest
from pytest import param

import ibis
import pandas as pd


@pytest.mark.parametrize('unit', [
    'Y', 'M', 'D',
    param('W', marks=pytest.mark.xfail),
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
    'Y', 'M', 'D',
    param('W', marks=pytest.mark.xfail)
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
    'Y', 'M', 'W', 'D', 'h', 'm', 's'
])
def test_integer_to_interval(backend, con, alltypes, df, unit):
    interval = alltypes.int_col.to_interval(unit=unit)
    expr = alltypes.timestamp_col + interval

    def convert_to_offset(x):
        resolution = '{}s'.format(interval.resolution)
        return pd.offsets.DateOffset(**{resolution: x})

    offset = df.int_col.apply(convert_to_offset)
    expected = df.timestamp_col + offset

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
