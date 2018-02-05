import pytest
from pytest import param

import pandas as pd

import ibis
import ibis.tests.util as tu


@pytest.mark.parametrize('attr', [
    'year', 'month', 'day',
])
@tu.skipif_unsupported
def test_date_extract(backend, alltypes, df, attr):
    expr = getattr(alltypes.timestamp_col.date(), attr)()
    expected = getattr(df.timestamp_col.dt, attr).astype('int32')

    result = expr.execute()
    expected = backend.default_series_rename(expected)

    backend.assert_series_equal(result, expected)


@pytest.mark.parametrize('attr', [
    'year', 'month', 'day',
    'hour', 'minute', 'second'
])
@tu.skipif_unsupported
def test_timestamp_extract(backend, alltypes, df, attr):
    expr = getattr(alltypes.timestamp_col, attr)()
    expected = getattr(df.timestamp_col.dt, attr).astype('int32')

    result = expr.execute()
    expected = backend.default_series_rename(expected)

    backend.assert_series_equal(result, expected)


@pytest.mark.parametrize('unit', [
    'Y', 'M', 'D',
    param('W', marks=pytest.mark.xfail),
    'h', 'm', 's', 'ms', 'us', 'ns'
])
@tu.skipif_unsupported
def test_timestamp_truncate(backend, alltypes, df, unit):
    expr = alltypes.timestamp_col.truncate(unit)

    dtype = 'datetime64[{}]'.format(unit)
    expected = pd.Series(df.timestamp_col.values.astype(dtype))

    result = expr.execute()
    expected = backend.default_series_rename(expected)

    backend.assert_series_equal(result, expected)


@pytest.mark.parametrize('unit', [
    'Y', 'M', 'D',
    param('W', marks=pytest.mark.xfail)
])
@tu.skipif_unsupported
def test_date_truncate(backend, alltypes, df, unit):
    expr = alltypes.timestamp_col.date().truncate(unit)

    dtype = 'datetime64[{}]'.format(unit)
    expected = pd.Series(df.timestamp_col.values.astype(dtype))

    result = expr.execute()
    expected = backend.default_series_rename(expected)

    backend.assert_series_equal(result, expected)


@pytest.mark.parametrize('unit', [
    'Y', 'M', 'W', 'D', 'h', 'm', 's'
])
@tu.skipif_unsupported
def test_integer_to_interval(backend, con, alltypes, df, unit):
    interval = alltypes.int_col.to_interval(unit=unit)
    expr = alltypes.timestamp_col + interval
    result = con.execute(expr)

    def convert_to_offset(x):
        resolution = '{}s'.format(interval.resolution)
        return pd.offsets.DateOffset(**{resolution: x})

    offset = df.int_col.apply(convert_to_offset)
    expected = df.timestamp_col + offset
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
@tu.skipif_unsupported
def test_timestamp_binop(backend, con, alltypes, df,
                         expr_fn, expected_fn):
    expr = expr_fn(alltypes)
    expected = expected_fn(df)

    result = con.execute(expr)
    expected = backend.default_series_rename(expected)
    backend.assert_series_equal(result, expected)


@pytest.mark.parametrize(('ibis_pattern', 'pandas_pattern'), [
    ('%Y%m%d', '%Y%m%d')
])
@tu.skipif_unsupported
def test_strftime(backend, con, alltypes, df, ibis_pattern, pandas_pattern):
    expr = alltypes.timestamp_col.strftime('%Y%m%d')
    expected = df.timestamp_col.dt.strftime('%Y%m%d')

    result = expr.execute()
    expected = backend.default_series_rename(expected)
    backend.assert_series_equal(result, expected)
