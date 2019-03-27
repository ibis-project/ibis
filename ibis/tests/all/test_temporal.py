import sys
import pytest
import warnings
from pytest import param

import numpy as np
import pandas as pd

import pandas.util.testing as tm

import ibis
import ibis.expr.datatypes as dt
import ibis.tests.util as tu

from ibis.tests.backends import MapD

from ibis.pandas.execution.temporal import day_name


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
@tu.skipif_backend(MapD)
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
    'Y', pytest.mark.xfail('Q'), 'M', 'W', 'D',
    'h', 'm', 's', pytest.mark.xfail('ms'), pytest.mark.xfail('us')
])
@tu.skipif_unsupported
def test_integer_to_interval_timestamp(backend, con, alltypes, df, unit):
    interval = alltypes.int_col.to_interval(unit=unit)
    expr = alltypes.timestamp_col + interval

    def convert_to_offset(x):
        resolution = '{}s'.format(interval.type().resolution)
        return pd.offsets.DateOffset(**{resolution: x})

    with warnings.catch_warnings():
        # both the implementation and test code raises pandas
        # PerformanceWarning, because We use DateOffset addition
        warnings.simplefilter('ignore')
        result = con.execute(expr)
        offset = df.int_col.apply(convert_to_offset)
        expected = df.timestamp_col + offset

    expected = backend.default_series_rename(expected)

    backend.assert_series_equal(result, expected)


@pytest.mark.parametrize('unit', ['Y', pytest.mark.xfail('Q'), 'M', 'W', 'D'])
@tu.skipif_unsupported
def test_integer_to_interval_date(backend, con, alltypes, df, unit):
    interval = alltypes.int_col.to_interval(unit=unit)
    array = alltypes.date_string_col.split('/')
    month, day, year = array[0], array[1], array[2]
    date_col = expr = ibis.literal('-').join([
        '20' + year, month, day
    ]).cast('date')
    expr = date_col + interval
    result = con.execute(expr)

    def convert_to_offset(x):
        resolution = '{}s'.format(interval.type().resolution)
        return pd.offsets.DateOffset(**{resolution: x})

    offset = df.int_col.apply(convert_to_offset)
    expected = pd.to_datetime(df.date_string_col) + offset
    expected = backend.default_series_rename(expected)

    backend.assert_series_equal(result, expected)


@pytest.mark.parametrize('unit', ['h', 'm', 's', 'ms', 'us'])
@tu.skipif_unsupported
def test_integer_to_interval_date_failure(backend, con, alltypes, df, unit):
    interval = alltypes.int_col.to_interval(unit=unit)
    array = alltypes.date_string_col.split('/')
    month, day, year = array[0], array[1], array[2]
    date_col = ibis.literal('-').join(['20' + year, month, day]).cast('date')
    with pytest.raises(TypeError):
        date_col + interval


date_value = pd.Timestamp('2017-12-31')
timestamp_value = pd.Timestamp('2018-01-01 18:18:18')


@pytest.mark.parametrize(('expr_fn', 'expected_fn'), [
    param(lambda t, be: t.timestamp_col + ibis.interval(days=4),
          lambda t, be: t.timestamp_col + pd.Timedelta(days=4),
          id='timestamp-add-interval'),
    param(lambda t, be: t.timestamp_col - ibis.interval(days=17),
          lambda t, be: t.timestamp_col - pd.Timedelta(days=17),
          id='timestamp-subtract-interval'),
    param(lambda t, be: t.timestamp_col.date() + ibis.interval(days=4),
          lambda t, be: t.timestamp_col.dt.floor('d') + pd.Timedelta(days=4),
          id='date-add-interval'),
    param(lambda t, be: t.timestamp_col.date() - ibis.interval(days=14),
          lambda t, be: t.timestamp_col.dt.floor('d') - pd.Timedelta(days=14),
          id='date-subtract-interval'),
    param(lambda t, be: t.timestamp_col - ibis.timestamp(timestamp_value),
          lambda t, be: pd.Series(
            t.timestamp_col.sub(timestamp_value).values.astype(
                'timedelta64[{}]'.format(be.returned_timestamp_unit))),
          id='timestamp-subtract-timestamp'),
    param(lambda t, be: t.timestamp_col.date() - ibis.date(date_value),
          lambda t, be: t.timestamp_col.dt.floor('d') - date_value,
          id='date-subtract-date'),
])
@tu.skipif_unsupported
def test_temporal_binop(backend, con, alltypes, df,
                        expr_fn, expected_fn):
    expr = expr_fn(alltypes, backend)
    expected = expected_fn(df, backend)

    result = con.execute(expr)
    expected = backend.default_series_rename(expected)

    backend.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    ('ibis_pattern', 'pandas_pattern'),
    [
        ('%Y%m%d', '%Y%m%d')
    ]
)
@tu.skipif_unsupported
def test_strftime(backend, con, alltypes, df, ibis_pattern, pandas_pattern):
    expr = alltypes.timestamp_col.strftime(ibis_pattern)
    expected = df.timestamp_col.dt.strftime(pandas_pattern)

    result = expr.execute()
    expected = backend.default_series_rename(expected)
    backend.assert_series_equal(result, expected)


unit_factors = {
    's': int(1e9),
    'ms': int(1e6),
    'us': int(1e3),
}


@pytest.mark.parametrize(
    'unit', ['D', 's', 'ms', pytest.mark.xfail('us'), pytest.mark.xfail('ns')]
)
@tu.skipif_unsupported
def test_to_timestamp(backend, con, alltypes, df, unit):
    if unit not in backend.supported_to_timestamp_units:
        pytest.skip(
            'Unit {!r} not supported by {} to_timestamp'.format(unit, backend))

    backend_unit = backend.returned_timestamp_unit
    factor = unit_factors[unit]

    ts = ibis.timestamp('2018-04-13 09:54:11.872832')
    pandas_ts = ibis.pandas.execute(ts).floor(unit).value

    # convert the now timestamp to the input unit being tested
    int_expr = ibis.literal(pandas_ts // factor)
    expr = int_expr.to_timestamp(unit)
    result = con.execute(expr)
    expected = pd.Timestamp(pandas_ts, unit='ns').floor(backend_unit)

    assert result == expected


@pytest.mark.parametrize(
    ('date', 'expected_index', 'expected_day'),
    [
        ('2017-01-01', 6, 'Sunday'),
        ('2017-01-02', 0, 'Monday'),
        ('2017-01-03', 1, 'Tuesday'),
        ('2017-01-04', 2, 'Wednesday'),
        ('2017-01-05', 3, 'Thursday'),
        ('2017-01-06', 4, 'Friday'),
        ('2017-01-07', 5, 'Saturday'),
    ]
)
@tu.skipif_unsupported
def test_day_of_week_scalar(backend, con, date, expected_index, expected_day):
    expr = ibis.literal(date).cast(dt.date)
    result_index = con.execute(expr.day_of_week.index())
    assert result_index == expected_index

    result_day = con.execute(expr.day_of_week.full_name())
    assert result_day.lower() == expected_day.lower()


@tu.skipif_unsupported
def test_day_of_week_column(backend, con, alltypes, df):
    expr = alltypes.timestamp_col.day_of_week

    result_index = expr.index().execute()
    expected_index = df.timestamp_col.dt.dayofweek.astype('int16')

    backend.assert_series_equal(
        result_index, expected_index, check_names=False)

    result_day = expr.full_name().execute()
    expected_day = day_name(df.timestamp_col.dt)

    backend.assert_series_equal(result_day, expected_day, check_names=False)


@pytest.mark.parametrize(
    ('day_of_week_expr', 'day_of_week_pandas'),
    [
        (
            lambda t: t.timestamp_col.day_of_week.index().count(),
            lambda s: s.dt.dayofweek.count(),
        ),
        (
            lambda t: t.timestamp_col.day_of_week.full_name().length().sum(),
            lambda s: day_name(s.dt).str.len().sum(),
        )
    ]
)
@tu.skipif_unsupported
def test_day_of_week_column_group_by(
    backend, con, alltypes, df, day_of_week_expr, day_of_week_pandas
):
    expr = alltypes.groupby('string_col').aggregate(
        day_of_week_result=day_of_week_expr
    )
    schema = expr.schema()
    assert schema['day_of_week_result'] == dt.int64

    result = expr.execute().sort_values('string_col')
    expected = df.groupby('string_col').timestamp_col.apply(
        day_of_week_pandas
    ).reset_index().rename(columns=dict(timestamp_col='day_of_week_result'))

    # FIXME(#1536): Pandas backend should use query.schema().apply_to
    backend.assert_frame_equal(
        result,
        expected,
        check_dtype=False,
        # python 2's handling of strings is annoying here wrt sqlalchemy's
        # column name string subclass
        check_column_type=sys.version_info.major != 2
    )


@tu.skipif_unsupported
@tu.skipif_backend(MapD)
def test_now(backend, con):
    expr = ibis.now()
    result = con.execute(expr)
    pandas_now = pd.Timestamp('now')
    assert isinstance(result, pd.Timestamp)

    # this could fail if we're testing in different timezones and we're testing
    # on Dec 31st
    assert result.year == pandas_now.year


@tu.skipif_unsupported
@tu.skipif_backend(MapD)
def test_now_from_projection(backend, con, alltypes, df):
    n = 5
    expr = alltypes[[ibis.now().name('ts')]].limit(n)
    result = expr.execute()
    ts = result.ts
    assert isinstance(result, pd.DataFrame)
    assert isinstance(ts, pd.Series)
    assert issubclass(ts.dtype.type, np.datetime64)
    assert len(result) == n
    assert ts.nunique() == 1

    now = pd.Timestamp('now')
    year_expected = pd.Series([now.year] * n, name='ts')
    tm.assert_series_equal(ts.dt.year, year_expected)
