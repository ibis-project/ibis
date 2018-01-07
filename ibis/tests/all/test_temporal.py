import pytest
from pytest import param

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


@pytest.mark.parametrize(('expr_fn', 'expected_fn'), [
    param(lambda t: t.int_col.to_interval(unit='Y'),
          lambda t: pd.to_timedelta(t.int_col, unit='Y'),
          id='integer-to-year-interval'),
    param(lambda t: t.int_col.to_interval(unit='W'),
          lambda t: pd.to_timedelta(t.int_col, unit='W'),
          id='integer-to-week-interval'),
    param(lambda t: t.int_col.to_interval(unit='M'),
          lambda t: pd.to_timedelta(t.int_col, unit='M'),
          id='integer-to-month-interval'),
    param(lambda t: t.int_col.to_interval(unit='D'),
          lambda t: pd.to_timedelta(t.int_col, unit='D'),
          id='integer-to-day-interval'),
])
def test_integer_to_interval(backend, con, alltypes, df,
                             expr_fn, expected_fn):
    expr = expr_fn(alltypes)
    expected = expected_fn(df)

    with backend.skip_unsupported():
        result = con.execute(expr)

    expected = backend.default_series_rename(expected)
    backend.assert_series_equal(result, expected)
